/**
 * Table Component
 *
 * Reusable, production-ready table component with sorting, filtering, and pagination.
 * Designed for institutional trading dashboard with high-performance requirements.
 */

import React, { useState, useMemo, useCallback } from 'react';

export interface Column<T> {
  key: string;
  label: string;
  render?: (row: T, index: number) => React.ReactNode;
  sortable?: boolean;
  align?: 'left' | 'center' | 'right';
  width?: string;
  className?: string;
}

export interface TableProps<T> {
  data: T[];
  columns: Column<T>[];
  keyExtractor: (row: T, index: number) => string | number;
  loading?: boolean;
  error?: Error | null;
  emptyMessage?: string;
  sortable?: boolean;
  filterable?: boolean;
  paginated?: boolean;
  pageSize?: number;
  onRowClick?: (row: T, index: number) => void;
  selectedRows?: Set<string | number>;
  onSelectionChange?: (selected: Set<string | number>) => void;
  className?: string;
  stickyHeader?: boolean;
}

function Table<T extends Record<string, any>>({
  data,
  columns,
  keyExtractor,
  loading = false,
  error = null,
  emptyMessage = 'No data available',
  sortable = true,
  filterable = true,
  paginated = true,
  pageSize = parseInt(process.env.REACT_APP_TABLE_PAGE_SIZE || '20'),
  onRowClick,
  selectedRows,
  onSelectionChange,
  className = '',
  stickyHeader = true
}: TableProps<T>) {
  const [sortConfig, setSortConfig] = useState<{
    key: string;
    direction: 'asc' | 'desc';
  } | null>(null);
  const [filterText, setFilterText] = useState('');
  const [currentPage, setCurrentPage] = useState(1);

  // Filtering
  const filteredData = useMemo(() => {
    if (!filterable || !filterText) return data;

    const lowerFilter = filterText.toLowerCase();
    return data.filter((row) => {
      return Object.values(row).some((value) => {
        if (value === null || value === undefined) return false;
        return String(value).toLowerCase().includes(lowerFilter);
      });
    });
  }, [data, filterText, filterable]);

  // Sorting
  const sortedData = useMemo(() => {
    if (!sortable || !sortConfig) return filteredData;

    const sorted = [...filteredData];
    sorted.sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];

      if (aValue === bValue) return 0;
      if (aValue === null || aValue === undefined) return 1;
      if (bValue === null || bValue === undefined) return -1;

      // Number comparison
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortConfig.direction === 'asc' ? aValue - bValue : bValue - aValue;
      }

      // String comparison
      const aStr = String(aValue).toLowerCase();
      const bStr = String(bValue).toLowerCase();

      if (sortConfig.direction === 'asc') {
        return aStr < bStr ? -1 : 1;
      } else {
        return aStr > bStr ? -1 : 1;
      }
    });

    return sorted;
  }, [filteredData, sortConfig, sortable]);

  // Pagination
  const paginatedData = useMemo(() => {
    if (!paginated) return sortedData;

    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    return sortedData.slice(startIndex, endIndex);
  }, [sortedData, currentPage, pageSize, paginated]);

  const totalPages = useMemo(() => {
    if (!paginated) return 1;
    return Math.ceil(sortedData.length / pageSize);
  }, [sortedData.length, pageSize, paginated]);

  // Sort handler
  const handleSort = useCallback((columnKey: string) => {
    setSortConfig((prev) => {
      if (prev?.key === columnKey) {
        return {
          key: columnKey,
          direction: prev.direction === 'asc' ? 'desc' : 'asc'
        };
      }
      return { key: columnKey, direction: 'asc' };
    });
  }, []);

  // Selection handler
  const handleSelectAll = useCallback((checked: boolean) => {
    if (!onSelectionChange) return;

    if (checked) {
      const allKeys = new Set(paginatedData.map((row, index) => keyExtractor(row, index)));
      onSelectionChange(allKeys);
    } else {
      onSelectionChange(new Set());
    }
  }, [paginatedData, keyExtractor, onSelectionChange]);

  const handleSelectRow = useCallback((rowKey: string | number, checked: boolean) => {
    if (!onSelectionChange || !selectedRows) return;

    const newSelection = new Set(selectedRows);
    if (checked) {
      newSelection.add(rowKey);
    } else {
      newSelection.delete(rowKey);
    }
    onSelectionChange(newSelection);
  }, [selectedRows, onSelectionChange]);

  // Pagination controls
  const goToPage = (page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  };

  // Render cell content
  const renderCell = (row: T, column: Column<T>, index: number) => {
    if (column.render) {
      return column.render(row, index);
    }
    const value = row[column.key];
    if (value === null || value === undefined) return '-';
    return String(value);
  };

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-700 rounded-lg p-6 text-center">
        <h3 className="text-red-400 font-semibold mb-2">Error Loading Data</h3>
        <p className="text-red-300 text-sm">{error.message}</p>
      </div>
    );
  }

  const showSelection = selectedRows !== undefined && onSelectionChange !== undefined;

  return (
    <div className={`bg-gray-800/50 border border-gray-700 rounded-lg ${className}`}>
      {/* Filter */}
      {filterable && (
        <div className="p-4 border-b border-gray-700">
          <input
            type="text"
            placeholder="Filter table..."
            value={filterText}
            onChange={(e) => {
              setFilterText(e.target.value);
              setCurrentPage(1);
            }}
            className="w-full md:w-64 bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className={`bg-gray-700/50 ${stickyHeader ? 'sticky top-0 z-10' : ''}`}>
            <tr>
              {showSelection && (
                <th className="px-4 py-3 text-left">
                  <input
                    type="checkbox"
                    checked={paginatedData.length > 0 && paginatedData.every((row, index) =>
                      selectedRows.has(keyExtractor(row, index))
                    )}
                    onChange={(e) => handleSelectAll(e.target.checked)}
                    className="rounded bg-gray-700 border-gray-600 text-blue-500"
                  />
                </th>
              )}
              {columns.map((column) => (
                <th
                  key={column.key}
                  className={`px-4 py-3 text-${column.align || 'left'} text-gray-300 font-semibold ${
                    column.sortable !== false && sortable ? 'cursor-pointer hover:text-white' : ''
                  } ${column.className || ''}`}
                  style={{ width: column.width }}
                  onClick={() => {
                    if (column.sortable !== false && sortable) {
                      handleSort(column.key);
                    }
                  }}
                >
                  <div className="flex items-center gap-2">
                    <span>{column.label}</span>
                    {column.sortable !== false && sortable && sortConfig?.key === column.key && (
                      <span className="text-blue-400">
                        {sortConfig.direction === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {loading && (
              <tr>
                <td colSpan={columns.length + (showSelection ? 1 : 0)} className="px-4 py-12 text-center">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent"></div>
                  <p className="text-gray-400 mt-2">Loading...</p>
                </td>
              </tr>
            )}
            {!loading && paginatedData.length === 0 && (
              <tr>
                <td colSpan={columns.length + (showSelection ? 1 : 0)} className="px-4 py-12 text-center text-gray-400">
                  {emptyMessage}
                </td>
              </tr>
            )}
            {!loading && paginatedData.map((row, index) => {
              const rowKey = keyExtractor(row, index);
              const isSelected = selectedRows?.has(rowKey) || false;

              return (
                <tr
                  key={rowKey}
                  className={`transition-colors ${
                    onRowClick ? 'cursor-pointer hover:bg-gray-700/30' : ''
                  } ${isSelected ? 'bg-blue-900/20' : ''}`}
                  onClick={() => onRowClick?.(row, index)}
                >
                  {showSelection && (
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={(e) => {
                          e.stopPropagation();
                          handleSelectRow(rowKey, e.target.checked);
                        }}
                        onClick={(e) => e.stopPropagation()}
                        className="rounded bg-gray-700 border-gray-600 text-blue-500"
                      />
                    </td>
                  )}
                  {columns.map((column) => (
                    <td
                      key={column.key}
                      className={`px-4 py-3 text-${column.align || 'left'} ${column.className || ''}`}
                    >
                      {renderCell(row, column, index)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {paginated && totalPages > 1 && (
        <div className="p-4 border-t border-gray-700 flex items-center justify-between">
          <div className="text-sm text-gray-400">
            Showing {((currentPage - 1) * pageSize) + 1} to {Math.min(currentPage * pageSize, sortedData.length)} of {sortedData.length} entries
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => goToPage(1)}
              disabled={currentPage === 1}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-gray-300 rounded transition-colors"
            >
              ««
            </button>
            <button
              onClick={() => goToPage(currentPage - 1)}
              disabled={currentPage === 1}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-gray-300 rounded transition-colors"
            >
              «
            </button>

            {/* Page numbers */}
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              let pageNum;
              if (totalPages <= 5) {
                pageNum = i + 1;
              } else if (currentPage <= 3) {
                pageNum = i + 1;
              } else if (currentPage >= totalPages - 2) {
                pageNum = totalPages - 4 + i;
              } else {
                pageNum = currentPage - 2 + i;
              }

              return (
                <button
                  key={pageNum}
                  onClick={() => goToPage(pageNum)}
                  className={`px-3 py-1 rounded transition-colors ${
                    currentPage === pageNum
                      ? 'bg-blue-600 text-white font-semibold'
                      : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}

            <button
              onClick={() => goToPage(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-gray-300 rounded transition-colors"
            >
              »
            </button>
            <button
              onClick={() => goToPage(totalPages)}
              disabled={currentPage === totalPages}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-gray-300 rounded transition-colors"
            >
              »»
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default Table;
