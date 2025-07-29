import React from 'react';

const LoadingSpinner = ({ size = 'medium', color = '#8b5cf6' }) => {
  const sizeClasses = {
    small: 'spinner-small',
    medium: 'spinner-medium',
    large: 'spinner-large'
  };

  return (
    <div className={`loading-spinner ${sizeClasses[size]}`}>
      <div 
        className="spinner-ring"
        style={{ borderTopColor: color }}
      ></div>
    </div>
  );
};

export default LoadingSpinner;