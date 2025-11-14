import React, { useEffect, useRef } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  showCloseButton?: boolean;
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
  footer?: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  size = 'md',
  showCloseButton = true,
  closeOnOverlayClick = true,
  closeOnEscape = true,
  footer,
}) => {
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent): void => {
      if (closeOnEscape && event.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, closeOnEscape, onClose]);

  useEffect(() => {
    if (isOpen && modalRef.current) {
      const focusableElements = modalRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const firstElement = focusableElements[0] as HTMLElement;
      if (firstElement) {
        firstElement.focus();
      }
    }
  }, [isOpen]);

  if (!isOpen) {
    return null;
  }

  const getSizeClasses = (): string => {
    switch (size) {
      case 'sm':
        return 'max-w-md';
      case 'md':
        return 'max-w-lg';
      case 'lg':
        return 'max-w-2xl';
      case 'xl':
        return 'max-w-4xl';
      case 'full':
        return 'max-w-full mx-4';
      default:
        return 'max-w-lg';
    }
  };

  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>): void => {
    if (closeOnOverlayClick && e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto"
      aria-labelledby="modal-title"
      role="dialog"
      aria-modal="true"
    >
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black bg-opacity-75 transition-opacity"
        onClick={handleOverlayClick}
      ></div>

      {/* Modal Container */}
      <div className="flex min-h-full items-center justify-center p-4 text-center sm:p-0">
        <div
          ref={modalRef}
          className={`relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all sm:my-8 sm:w-full ${getSizeClasses()}`}
        >
          {/* Header */}
          {(title || showCloseButton) && (
            <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
              <div className="flex items-center justify-between">
                {title && (
                  <h3
                    className="text-lg font-semibold text-white"
                    id="modal-title"
                  >
                    {title}
                  </h3>
                )}
                {showCloseButton && (
                  <button
                    type="button"
                    className="ml-auto text-gray-400 hover:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-lg p-1"
                    onClick={onClose}
                    aria-label="Close modal"
                  >
                    <XMarkIcon className="h-6 w-6" />
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Content */}
          <div className="bg-gray-800 px-6 py-4">
            {children}
          </div>

          {/* Footer */}
          {footer && (
            <div className="bg-gray-800 border-t border-gray-700 px-6 py-4">
              {footer}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Modal;

// Convenience components for common modal patterns

interface ConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  confirmButtonClass?: string;
  loading?: boolean;
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  confirmButtonClass = 'bg-blue-600 hover:bg-blue-700',
  loading = false,
}) => {
  const handleConfirm = (): void => {
    onConfirm();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      size="sm"
      footer={
        <div className="flex justify-end space-x-3">
          <button
            type="button"
            onClick={onClose}
            disabled={loading}
            className="px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-600 transition-colors disabled:opacity-50"
          >
            {cancelText}
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            disabled={loading}
            className={`px-4 py-2 text-white rounded-md transition-colors disabled:opacity-50 ${confirmButtonClass}`}
          >
            {loading ? (
              <span className="flex items-center">
                <svg
                  className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Processing...
              </span>
            ) : (
              confirmText
            )}
          </button>
        </div>
      }
    >
      <p className="text-gray-300">{message}</p>
    </Modal>
  );
};

interface AlertModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  message: string;
  type?: 'info' | 'success' | 'warning' | 'error';
  buttonText?: string;
}

export const AlertModal: React.FC<AlertModalProps> = ({
  isOpen,
  onClose,
  title,
  message,
  type = 'info',
  buttonText = 'OK',
}) => {
  const getIconColor = (): string => {
    switch (type) {
      case 'success':
        return 'text-green-500';
      case 'warning':
        return 'text-yellow-500';
      case 'error':
        return 'text-red-500';
      default:
        return 'text-blue-500';
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      size="sm"
      footer={
        <div className="flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            {buttonText}
          </button>
        </div>
      }
    >
      <div className="flex items-start">
        <div className={`flex-shrink-0 ${getIconColor()}`}>
          <svg
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            {type === 'success' && (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            )}
            {type === 'warning' && (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            )}
            {type === 'error' && (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            )}
            {type === 'info' && (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            )}
          </svg>
        </div>
        <div className="ml-3">
          <p className="text-sm text-gray-300">{message}</p>
        </div>
      </div>
    </Modal>
  );
};
