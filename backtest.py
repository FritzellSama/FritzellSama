"""
Backtesting Engine - Quantum Trader Pro
Système de backtesting pour tester les stratégies sur données historiques
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from config import ConfigLoader
from core.binance_client import BinanceClient
from data.data_loader import DataLoader
from strategies.strategy_manager import StrategyManager
from risk.position_sizer import PositionSizer
from risk.stop_loss_manager import StopLossManager
from risk.take_profit_manager import TakeProfitManager
from utils.logger import setup_logger

class BacktestEngine:
    """
    Moteur de backtesting qui:
    - Charge données historiques
    - Simule l'exécution des stratégies
    - Calcule les métriques de performance
    - Génère un rapport détaillé
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le moteur de backtesting

        Args:
            config_path: Chemin vers config.yaml
        """

        self.logger = setup_logger('BacktestEngine')
        self.logger.info("=" * 70)
        self.logger.info("📈 QUANTUM TRADER PRO - BACKTESTING")
        self.logger.info("=" * 70)

        # Charger config
        try:
            self.config_loader = ConfigLoader(config_path)
            self.config = self.config_loader.config
            self.backtest_config = self.config.get('backtest', {})
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement config: {e}")
            sys.exit(1)

        # Paramètres backtest
        data_config = self.backtest_config.get('data', {})
        self.start_date = data_config.get('start_date', '2023-01-01')
        self.end_date = data_config.get('end_date', '2024-11-08')
        self.warmup_bars = data_config.get('warmup_bars', 100)

        sim_config = self.backtest_config.get('simulation', {})
        self.initial_balance = sim_config.get('initial_balance', 300)
        self.commission_maker = sim_config.get('commission_maker', 0.1) / 100
        self.commission_taker = sim_config.get('commission_taker', 0.1) / 100

        # État
        self.balance = self.initial_balance
        self.equity_curve = []
        self.trades = []
        self.positions = []

        # Initialiser composants
        self._initialize_components()

    def _initialize_components(self):
        """Initialise les composants nécessaires"""

        try:
            # Client (pour data uniquement)
            self.client = BinanceClient(self.config)

            # Data loader
            self.data_loader = DataLoader(self.client, self.config)

            # Strategy manager
            self.strategy_manager = StrategyManager(self.config)

            # Risk managers
            self.position_sizer = PositionSizer(self.config)
            self.stop_loss_manager = StopLossManager(self.config)
            self.take_profit_manager = TakeProfitManager(self.config)

            self.logger.info("✅ Composants initialisés")

        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation: {e}")
            raise

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Charge les données historiques pour le backtest

        Returns:
            Dict {timeframe: DataFrame}
        """

        self.logger.info(f"📥 Chargement données: {self.start_date} → {self.end_date}")

        symbol = self.config['symbols']['primary']

        # Charger multi-timeframe
        data = {}
        for tf in ['1h', '5m']:
            df = self.data_loader.load_historical_data(
                symbol=symbol,
                timeframe=tf,
                limit=5000,
                start_date=self.start_date,
                end_date=self.end_date
            )
            if not df.empty:
                data[tf] = df

        if not data:
            self.logger.error("❌ Aucune donnée chargée")
            return {}

        # Logger les données chargées
        for tf, df in data.items():
            self.logger.info(f"✅ {len(df)} bougies {tf} chargées")

        return data

    def run(self) -> Dict:
        """
        Exécute le backtest complet

        Returns:
            Dict avec résultats et métriques
        """

        # Charger données
        data = self.load_data()

        if not data or '5m' not in data:
            self.logger.error("❌ Données insuffisantes")
            return {}

        # Prendre timeframe principal (5m)
        df_5m = data['5m']
        df_1h = data.get('1h', df_5m.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }))

        self.logger.info(f"🔄 Démarrage backtest sur {len(df_5m)} bougies")

        # Boucle principale
        for i in range(self.warmup_bars, len(df_5m)):
            current_time = df_5m.index[i]
            current_price = df_5m.iloc[i]['close']

            # Données jusqu'à maintenant
            historical_5m = df_5m.iloc[:i+1]
            historical_1h = df_1h[df_1h.index <= current_time]

            # 1. Mettre à jour positions ouvertes
            self._update_positions(current_price, df_5m.iloc[i])

            # 2. Générer signaux
            try:
                market_data = {
                    '5m': historical_5m,
                    '1h': historical_1h,
                    'ticker': {'last': current_price},
                    'orderbook': {}
                }

                signals = self.strategy_manager.generate_all_signals(market_data)

                # 3. Exécuter signaux - FIX: Itérer correctement sur le dict
                for strategy_name, strategy_signals in signals.items():
                    for signal in strategy_signals:
                        self._execute_signal(signal, current_price, current_time)

            except Exception as e:
                self.logger.debug(f"⚠️ Erreur à {current_time}: {e}")

            # 4. Enregistrer equity
            total_equity = self._calculate_equity(current_price)
            self.equity_curve.append({
                'timestamp': current_time,
                'balance': self.balance,
                'equity': total_equity,
                'num_positions': len(self.positions)
            })

            # Log progrès
            if i % 1000 == 0:
                progress = (i / len(df_5m)) * 100
                self.logger.info(
                    f"📊 Progrès: {progress:.1f}% | "
                    f"Balance: ${self.balance:.2f} | "
                    f"Trades: {len(self.trades)}"
                )

        # Fermer positions restantes
        if not df_5m.empty:
            self._close_all_positions(df_5m.iloc[-1]['close'])
        else:
            self.logger.error("❌ Impossible de fermer les positions : df_5m est vide")

        # Calculer métriques
        results = self._calculate_metrics()

        # Afficher résultats
        self._print_results(results)

        return results

    def _execute_signal(self, signal, current_price: float, current_time):
        """Exécute un signal en backtest"""

        # Vérifier si on peut trader
        max_positions = self.config['risk']['max_positions_simultaneous']
        if len(self.positions) >= max_positions:
            self.logger.debug(
                f"⚠️ Max positions atteint: {len(self.positions)}/{max_positions}"
            )
            return

        # FIX: Calculer SL/TP AVANT position_sizer
        if signal.stop_loss:
            stop_loss = signal.stop_loss
        else:
            atr = self._calculate_atr_simple(current_price)
            # Utiliser create_stop_loss() avec un position_id temporaire
            position_id = f"backtest_{current_time.strftime('%Y%m%d_%H%M%S')}_{signal.strategy}"
            stop_loss = self.stop_loss_manager.create_stop_loss(
                position_id=position_id,
                entry_price=current_price,
                side=signal.type,
                atr=atr
            )

        # Calculer taille position (maintenant avec stop_loss valide)
        size = self.position_sizer.calculate_position_size(
            capital=self.balance,
            entry_price=current_price,
            stop_loss=stop_loss,
            confidence=signal.confidence,
            signal_type=signal.type
        )

        if size == 0:
            self.logger.warning(
                f"⚠️ Taille position = 0 | "
                f"Balance: ${self.balance:.2f} | "
                f"Price: ${current_price:.2f} | "
                f"SL: {stop_loss} | "
                f"Confidence: {signal.confidence}"
            )
            return

        # Calculer commission
        cost = size * current_price
        commission = cost * self.commission_taker

        if cost + commission > self.balance:
            self.logger.debug(f"⚠️ Pas assez de capital: {cost + commission:.2f} > {self.balance:.2f}")
            return  # Pas assez de capital

        # Calculer take profit
        take_profit_levels = self.take_profit_manager.calculate_take_profit_levels(
            entry_price=current_price,
            stop_loss=stop_loss,
            position_side=signal.type
        )

        # Créer position
        position = {
            'id': len(self.positions),
            'symbol': signal.symbol,
            'side': signal.type,
            'entry_price': current_price,
            'entry_time': current_time,
            'size': size,
            'initial_size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit_levels,
            'strategy': signal.strategy,
            'commission_paid': commission
        }

        # Déduire du balance
        self.balance -= (cost + commission)

        # Ajouter position
        self.positions.append(position)

        self.logger.info(
            f"📝 Position ouverte: {signal.type.upper()} @ ${current_price:.2f} | "
            f"Size: {size:.8f} BTC (${cost:.2f}) | SL: ${stop_loss:.2f}"
        )

    def _update_positions(self, current_price: float, current_bar: pd.Series):
        """Met à jour les positions ouvertes"""

        positions_to_close = []

        for pos in self.positions:
            # Vérifier stop-loss
            if pos['side'] == 'long':
                if current_price <= pos['stop_loss']:
                    positions_to_close.append((pos, pos['stop_loss'], 'stop_loss'))
                    continue
            else:  # short
                if current_price >= pos['stop_loss']:
                    positions_to_close.append((pos, pos['stop_loss'], 'stop_loss'))
                    continue

            # Vérifier take-profit
            for tp in pos['take_profit']:
                if tp.get('filled', False):
                    continue

                tp_price = tp['price']

                if pos['side'] == 'long':
                    if current_price >= tp_price:
                        positions_to_close.append((pos, tp_price, f"tp_{tp['level']}"))
                        tp['filled'] = True
                        break
                else:  # short
                    if current_price <= tp_price:
                        positions_to_close.append((pos, tp_price, f"tp_{tp['level']}"))
                        tp['filled'] = True
                        break

        # Fermer positions
        for pos, exit_price, reason in positions_to_close:
            self._close_position(pos, exit_price, current_bar.name, reason)

    def _close_position(self, position: Dict, exit_price: float, exit_time, reason: str):
        """Ferme une position"""

        # Calculer PnL
        if position['side'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['size']

        # Commission de sortie
        cost = position['size'] * exit_price
        commission = cost * self.commission_taker

        # PnL net
        pnl_net = pnl - commission - position['commission_paid']

        # Ajouter au balance
        self.balance += cost + pnl_net + position['commission_paid']  # Rembourser commission entrée

        # Enregistrer trade
        duration = (exit_time - position['entry_time']).total_seconds() / 60

        trade = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'symbol': position['symbol'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl_net,
            'pnl_percent': (pnl_net / (position['entry_price'] * position['size'])) * 100,
            'duration_minutes': duration,
            'reason': reason,
            'strategy': position['strategy']
        }

        self.trades.append(trade)

        # Retirer position
        self.positions.remove(position)

        self.logger.info(
            f"{'✅' if pnl_net > 0 else '❌'} Position fermée: "
            f"{position['side'].upper()} | PnL: ${pnl_net:.2f} ({trade['pnl_percent']:.2f}%) | Raison: {reason}"
        )

    def _close_all_positions(self, current_price: float):
        """Ferme toutes les positions restantes"""

        current_time = datetime.now()

        for pos in self.positions[:]:  # Copy list
            self._close_position(pos, current_price, current_time, 'end_of_backtest')

    def _calculate_equity(self, current_price: float) -> float:
        """Calcule l'equity totale (balance + positions)"""

        positions_value = 0

        for pos in self.positions:
            if pos['side'] == 'long':
                pnl = (current_price - pos['entry_price']) * pos['size']
            else:
                pnl = (pos['entry_price'] - current_price) * pos['size']

            positions_value += pnl

        return self.balance + positions_value

    def _calculate_atr_simple(self, current_price: float) -> float:
        """Calcule ATR simple (estimation)"""
        return current_price * 0.02  # 2% du prix

    def _calculate_metrics(self) -> Dict:
        """Calcule toutes les métriques de performance"""

        if not self.trades:
            return {}

        df_trades = pd.DataFrame(self.trades)
        df_equity = pd.DataFrame(self.equity_curve)

        # Métriques de base
        total_trades = len(self.trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])

        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # PnL
        total_pnl = df_trades['pnl'].sum()
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100

        # Moyennes
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        # Profit factor
        total_wins = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        total_losses = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Drawdown
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak'] * 100
        max_drawdown = df_equity['drawdown'].min()

        # Sharpe Ratio (simplifié)
        returns = df_equity['equity'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 12) if returns.std() > 0 else 0

        # Durée moyenne
        avg_duration = df_trades['duration_minutes'].mean()

        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_duration_minutes': avg_duration,
            'expectancy': expectancy
        }

    def _print_results(self, results: Dict):
        """Affiche les résultats du backtest"""

        self.logger.info("=" * 70)
        self.logger.info("📊 RÉSULTATS DU BACKTEST")
        self.logger.info("=" * 70)

        if not results or 'initial_balance' not in results:
            self.logger.error("❌ Aucun résultat à afficher - Pas de trades exécutés")
            self.logger.info("=" * 70)
            return

        self.logger.info(f"💰 Balance initiale: ${results['initial_balance']:.2f}")
        self.logger.info(f"💰 Balance finale: ${results['final_balance']:.2f}")
        self.logger.info(f"📈 PnL Total: ${results['total_pnl']:.2f} ({results['total_return_pct']:.2f}%)")
        self.logger.info("")

        self.logger.info(f"📊 Total trades: {results['total_trades']}")
        self.logger.info(f"✅ Winning: {results['winning_trades']}")
        self.logger.info(f"❌ Losing: {results['losing_trades']}")
        self.logger.info(f"🎯 Win Rate: {results['win_rate']:.2f}%")
        self.logger.info("")

        self.logger.info(f"💵 Avg Win: ${results['avg_win']:.2f}")
        self.logger.info(f"💸 Avg Loss: ${results['avg_loss']:.2f}")
        self.logger.info(f"⚡ Profit Factor: {results['profit_factor']:.2f}")
        self.logger.info(f"📉 Max Drawdown: {results['max_drawdown']:.2f}%")
        self.logger.info(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        self.logger.info(f"⏱️  Avg Duration: {results['avg_duration_minutes']:.1f} min")
        self.logger.info(f"🎲 Expectancy: ${results['expectancy']:.2f}")

        self.logger.info("=" * 70)

def main():
    """Point d'entrée pour backtesting"""

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║            📈 QUANTUM TRADER PRO - BACKTESTING 📈                ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    try:
        engine = BacktestEngine()
        results = engine.run()

        if not results:
            print("❌ Backtest échoué")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Interruption utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
