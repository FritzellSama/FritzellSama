import React, { useState } from 'react';
import PositionList from '../components/trading/PositionList';
import PnLChart from '../components/charts/PnLChart';
import PerformanceMetrics from '../components/analytics/PerformanceMetrics';
import {
  ChartBarIcon,
  CurrencyDollarIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline';

interface TabItem {
  id: string;
  name: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
}

const tabs: TabItem[] = [
  { id: 'positions', name: 'Positions', icon: ChartBarIcon },
  { id: 'performance', name: 'Performance', icon: ArrowTrendingUpIcon },
  { id: 'pnl', name: 'P&L Chart', icon: CurrencyDollarIcon },
];

const Positions: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('positions');

  const renderTabContent = (): React.ReactNode => {
    switch (activeTab) {
      case 'positions':
        return (
          <div className="space-y-6">
            <PositionList />
          </div>
        );
      case 'performance':
        return (
          <div className="space-y-6">
            <PerformanceMetrics />
          </div>
        );
      case 'pnl':
        return (
          <div className="space-y-6">
            <PnLChart height={500} />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Positions & Performance</h1>
        <p className="mt-2 text-sm text-gray-400">
          Monitor your open positions and track performance metrics
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-700">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  group inline-flex items-center py-4 px-1 border-b-2 font-medium text-sm
                  ${
                    isActive
                      ? 'border-blue-500 text-blue-500'
                      : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300'
                  }
                  transition-colors
                `}
              >
                <Icon
                  className={`
                    -ml-0.5 mr-2 h-5 w-5
                    ${isActive ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-300'}
                  `}
                />
                {tab.name}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div>{renderTabContent()}</div>
    </div>
  );
};

export default Positions;
