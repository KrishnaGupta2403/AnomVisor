import React from 'react';
import ImageDisplay from './ImageDisplay';
import './ModelResults.css';

function ModelResults({ result }) {
  if (!result) return null;

  // Convert result data to image display format
  const images = result.images ? result.images.map((img, index) => ({
    url: img.url || img.path,
    name: img.name || `Result ${index + 1}`,
    description: img.description
  })) : [];

  // Format classification report for display
  const formatClassificationReport = (report) => {
    if (!report) return null;
    
    const rows = [];
    // Add per-class metrics
    Object.entries(report).forEach(([key, value]) => {
      if (typeof value === 'object' && value !== null && !['accuracy', 'macro avg', 'weighted avg'].includes(key)) {
        rows.push(
          <tr key={key}>
            <td>Class {key}</td>
            <td>{value.precision?.toFixed(4) || 'N/A'}</td>
            <td>{value.recall?.toFixed(4) || 'N/A'}</td>
            <td>{value.f1_score?.toFixed(4) || 'N/A'}</td>
            <td>{value.support || 'N/A'}</td>
          </tr>
        );
      }
    });

    // Add averages
    if (report['macro avg']) {
      rows.push(
        <tr key="macro-avg">
          <td>Macro Average</td>
          <td>{report['macro avg'].precision?.toFixed(4) || 'N/A'}</td>
          <td>{report['macro avg'].recall?.toFixed(4) || 'N/A'}</td>
          <td>{report['macro avg'].f1_score?.toFixed(4) || 'N/A'}</td>
          <td>{report['macro avg'].support || 'N/A'}</td>
        </tr>
      );
    }

    if (report['weighted avg']) {
      rows.push(
        <tr key="weighted-avg">
          <td>Weighted Average</td>
          <td>{report['weighted avg'].precision?.toFixed(4) || 'N/A'}</td>
          <td>{report['weighted avg'].recall?.toFixed(4) || 'N/A'}</td>
          <td>{report['weighted avg'].f1_score?.toFixed(4) || 'N/A'}</td>
          <td>{report['weighted avg'].support || 'N/A'}</td>
        </tr>
      );
    }

    return rows;
  };

  return (
    <div className="model-results">
      <h2>Analysis Results</h2>
      
      {/* Display any numerical results or statistics */}
      {result.stats && (
        <div className="results-stats">
          <h3>Statistics</h3>
          <div className="stats-section">
            <h4>Overall Accuracy</h4>
            <p>{result.stats.accuracy ? (result.stats.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
          </div>

          {result.stats.classification_report && (
            <div className="stats-section">
              <h4>Classification Report</h4>
              <table>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {formatClassificationReport(result.stats.classification_report)}
                </tbody>
              </table>
            </div>
          )}

          {result.stats.confusion_matrix && (
            <div className="stats-section">
              <h4>Confusion Matrix</h4>
              <table>
                <tbody>
                  {result.stats.confusion_matrix.map((row, i) => (
                    <tr key={i}>
                      {row.map((cell, j) => (
                        <td key={j}>{cell}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Display output images */}
      {images.length > 0 && (
        <ImageDisplay
          images={images}
          title="Analysis Output Images"
        />
      )}

      {/* Display any additional information */}
      {result.info && (
        <div className="results-info">
          <h3>Additional Information</h3>
          <p>{result.info}</p>
        </div>
      )}
    </div>
  );
}

export default ModelResults;
