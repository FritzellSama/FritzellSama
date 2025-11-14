import React, { useState, useEffect } from 'react';
import {
  CpuChipIcon,
  PlayIcon,
  StopIcon,
  ArrowPathIcon,
  ChartBarIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';

interface MLModel {
  model_id: string;
  name: string;
  type: string;
  status: 'TRAINING' | 'READY' | 'STOPPED' | 'ERROR';
  accuracy: string;
  precision: string;
  recall: string;
  f1_score: string;
  last_trained: string;
  predictions_count: number;
  version: string;
}

interface TrainingMetrics {
  epoch: number;
  loss: string;
  val_loss: string;
  accuracy: string;
  val_accuracy: string;
}

interface ModelMetrics {
  model_id: string;
  metrics: {
    [key: string]: string;
  };
  training_history: TrainingMetrics[];
}

const MLModels: React.FC = () => {
  const [models, setModels] = useState<MLModel[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [refreshing, setRefreshing] = useState<boolean>(false);

  useEffect(() => {
    fetchModels();

    const interval = setInterval(() => {
      fetchModels(true);
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (selectedModel) {
      fetchModelMetrics(selectedModel);
    }
  }, [selectedModel]);

  const fetchModels = async (background: boolean = false): Promise<void> => {
    if (!background) {
      setLoading(true);
    }
    setError('');

    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/api/v1/ml/models`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }

      const data = await response.json();
      setModels(data.models || []);
    } catch (err) {
      console.error('Error fetching ML models:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load ML models';
      setError(errorMessage);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const fetchModelMetrics = async (modelId: string): Promise<void> => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/api/v1/ml/models/${modelId}/metrics`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch model metrics: ${response.statusText}`);
      }

      const data = await response.json();
      setModelMetrics(data);
    } catch (err) {
      console.error('Error fetching model metrics:', err);
    }
  };

  const handleStartModel = async (modelId: string): Promise<void> => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/api/v1/ml/models/${modelId}/start`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to start model: ${response.statusText}`);
      }

      await fetchModels();
    } catch (err) {
      console.error('Error starting model:', err);
      setError(err instanceof Error ? err.message : 'Failed to start model');
    }
  };

  const handleStopModel = async (modelId: string): Promise<void> => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/api/v1/ml/models/${modelId}/stop`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to stop model: ${response.statusText}`);
      }

      await fetchModels();
    } catch (err) {
      console.error('Error stopping model:', err);
      setError(err instanceof Error ? err.message : 'Failed to stop model');
    }
  };

  const handleRetrainModel = async (modelId: string): Promise<void> => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/api/v1/ml/models/${modelId}/retrain`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to retrain model: ${response.statusText}`);
      }

      await fetchModels();
    } catch (err) {
      console.error('Error retraining model:', err);
      setError(err instanceof Error ? err.message : 'Failed to retrain model');
    }
  };

  const handleRefresh = (): void => {
    setRefreshing(true);
    fetchModels();
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'READY':
        return 'text-green-400 bg-green-400/10 border-green-400/20';
      case 'TRAINING':
        return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      case 'STOPPED':
        return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
      case 'ERROR':
        return 'text-red-400 bg-red-400/10 border-red-400/20';
      default:
        return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
    }
  };

  const formatDate = (dateString: string): string => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch {
      return dateString;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">ML Models</h1>
          <p className="mt-1 text-sm text-gray-400">
            Manage and monitor machine learning models
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
        >
          <ArrowPathIcon className={`h-5 w-5 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-500 bg-opacity-10 border border-red-500 rounded-md p-4 flex items-start">
          <ExclamationCircleIcon className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
          <span className="text-sm text-red-400">{error}</span>
        </div>
      )}

      {/* Models Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {models.map((model) => (
          <div
            key={model.model_id}
            className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:border-blue-500 transition-colors cursor-pointer"
            onClick={() => setSelectedModel(model.model_id)}
          >
            {/* Model Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center">
                <CpuChipIcon className="h-8 w-8 text-blue-500" />
                <div className="ml-3">
                  <h3 className="text-lg font-semibold text-white">{model.name}</h3>
                  <p className="text-sm text-gray-400">{model.type}</p>
                </div>
              </div>
              <span
                className={`px-2 py-1 text-xs font-semibold rounded border ${getStatusColor(
                  model.status
                )}`}
              >
                {model.status}
              </span>
            </div>

            {/* Model Metrics */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-xs text-gray-400">Accuracy</p>
                <p className="text-lg font-semibold text-white">{model.accuracy}%</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Precision</p>
                <p className="text-lg font-semibold text-white">{model.precision}%</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Recall</p>
                <p className="text-lg font-semibold text-white">{model.recall}%</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">F1 Score</p>
                <p className="text-lg font-semibold text-white">{model.f1_score}</p>
              </div>
            </div>

            {/* Model Info */}
            <div className="space-y-2 mb-4 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Version:</span>
                <span className="text-white">{model.version}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Predictions:</span>
                <span className="text-white">{model.predictions_count.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Last Trained:</span>
                <span className="text-white">{formatDate(model.last_trained)}</span>
              </div>
            </div>

            {/* Actions */}
            <div className="flex space-x-2">
              {model.status === 'STOPPED' ? (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleStartModel(model.model_id);
                  }}
                  className="flex-1 flex items-center justify-center px-3 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-sm"
                >
                  <PlayIcon className="h-4 w-4 mr-1" />
                  Start
                </button>
              ) : (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleStopModel(model.model_id);
                  }}
                  className="flex-1 flex items-center justify-center px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors text-sm"
                  disabled={model.status === 'TRAINING'}
                >
                  <StopIcon className="h-4 w-4 mr-1" />
                  Stop
                </button>
              )}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRetrainModel(model.model_id);
                }}
                className="flex-1 flex items-center justify-center px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm"
                disabled={model.status === 'TRAINING'}
              >
                <ChartBarIcon className="h-4 w-4 mr-1" />
                Retrain
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Empty State */}
      {models.length === 0 && !error && (
        <div className="text-center py-12">
          <CpuChipIcon className="mx-auto h-12 w-12 text-gray-600" />
          <h3 className="mt-2 text-sm font-medium text-gray-400">No ML models</h3>
          <p className="mt-1 text-sm text-gray-500">
            No machine learning models are currently configured.
          </p>
        </div>
      )}
    </div>
  );
};

export default MLModels;
