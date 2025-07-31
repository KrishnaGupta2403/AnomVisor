import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ModelResults from './components/ModelResults';
import DatasetSelector from './components/DatasetSelector';
import Toast from './components/Toast';

function App() {
  // State management
  const [result, setResult] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [backendStatus, setBackendStatus] = useState('Checking backend status...');
  const [toast, setToast] = useState(null);
  const [darkMode, setDarkMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);

  // Toast management
  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type });
  }, []);

  const handleToastClose = useCallback(() => {
    setToast(null);
  }, []);

  // Help and Shortcuts toggles
  const toggleHelp = useCallback(() => {
    setShowHelp(prev => !prev);
    setShowShortcuts(false);
    showToast(showHelp ? 'Help panel closed' : 'Help panel opened', 'info');
  }, [showHelp, showToast]);

  const toggleShortcuts = useCallback(() => {
    setShowShortcuts(prev => !prev);
    setShowHelp(false);
    showToast(showShortcuts ? 'Shortcuts panel closed' : 'Shortcuts panel opened', 'info');
  }, [showShortcuts, showToast]);

  // Dark mode toggle
  const toggleDarkMode = useCallback(() => {
    setDarkMode(prev => !prev);
    showToast(`Switched to ${darkMode ? 'light' : 'dark'} mode`, 'info');
  }, [darkMode, showToast]);

  // Load theme from localStorage on initial render
  useEffect(() => {
    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme !== null) {
      setDarkMode(JSON.parse(savedTheme));
    }
  }, []);

  // Apply and persist dark mode
  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event) => {
      // Ctrl/Cmd + D to toggle dark mode
      if ((event.ctrlKey || event.metaKey) && event.key === 'd') {
        event.preventDefault();
        toggleDarkMode();
      }
      // Escape to close toast
      if (event.key === 'Escape' && toast) {
        handleToastClose();
      }
      // Ctrl/Cmd + H to toggle help
      if ((event.ctrlKey || event.metaKey) && event.key === 'h') {
        event.preventDefault();
        toggleHelp();
      }
      // Ctrl/Cmd + S to toggle shortcuts
      if ((event.ctrlKey || event.metaKey) && event.key === 's') {
        event.preventDefault();
        toggleShortcuts();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [toast, toggleDarkMode, handleToastClose, toggleHelp, toggleShortcuts]);

  // Accessibility announcements
  const announceStatus = useCallback((message) => {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    document.body.appendChild(announcement);
    setTimeout(() => announcement.remove(), 1000);
  }, []);

  // Check backend status with retry mechanism
  const checkBackendStatus = useCallback(async () => {
    let retries = 3;
    while (retries > 0) {
      try {
        const response = await fetch('http://127.0.0.1:5000/ping');
        const data = await response.json();
        if (response.ok) {
          setBackendStatus(data.message);
          announceStatus('Backend connection successful');
          return;
        }
      } catch (error) {
        retries--;
        if (retries === 0) {
          setBackendStatus('Error connecting to backend');
          announceStatus('Backend connection failed');
          showToast('Backend connection failed. Please check your connection.', 'error');
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }, [showToast, announceStatus]);

  useEffect(() => {
    checkBackendStatus();
    // Set up periodic backend status check
    const intervalId = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(intervalId);
  }, [checkBackendStatus]);

  const handleUploadSuccess = (results) => {
    setResult(results);
    showToast('Upload successful! Results are ready.', 'success');
  };

  const handleUploadFailure = (error) => {
    showToast(error?.message || 'Upload failed. Please try again.', 'error');
  };

  const handleDatasetChange = (dataset) => {
    setSelectedDataset(dataset);
    showToast(`Dataset changed to: ${dataset}`, 'info');
  };

  return (
    <div className={`App ${darkMode ? 'dark' : 'light'}`}>
      <header>
        <h1>Hyperspectral Image Anomaly Detection</h1>
        <div className="header-info">
          <p className="status">
            <span className={`status-indicator ${backendStatus.includes('Error') ? 'error' : 'success'}`}></span>
            Backend status: {backendStatus}
          </p>
          <div className="system-info">
            <span className="separator">•</span>
            <span className="last-check">Last checked: {new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </header>

      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={handleToastClose}
        />
      )}

      <div className="container">
        <main>
          <DatasetSelector
            selectedDataset={selectedDataset}
            onDatasetChange={handleDatasetChange}
          />
          <FileUpload
            selectedDataset={selectedDataset}
            onUploadSuccess={handleUploadSuccess}
            onUploadFailure={handleUploadFailure}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
          {isLoading ? (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <p>Processing your request...</p>
            </div>
          ) : (
            <ModelResults result={result} />
          )}
        </main>
      </div>

      <footer>
        <div className="footer-content">
          <div className="footer-actions">
            <div className="footer-buttons">
              <button 
                className="help-btn" 
                onClick={toggleHelp}
                aria-label="Help"
                title="Help (Ctrl/Cmd + H)"
              >
                Help
              </button>
              <button 
                className="shortcuts-btn" 
                onClick={toggleShortcuts}
                aria-label="Keyboard Shortcuts"
                title="Keyboard Shortcuts (Ctrl/Cmd + S)"
              >
                Shortcuts
              </button>
            </div>
            <button 
              className="toggle-dark-mode-btn" 
              onClick={toggleDarkMode}
              aria-label={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
              title="Toggle Dark Mode (Ctrl/Cmd + D)"
            >
              {darkMode ? '🌞 Switch to Light Mode' : '🌙 Switch to Dark Mode'}
            </button>
          </div>
          <div className="team-info">
            <p className="team-name">AnomVisor</p>
            <p className="copyright">© {new Date().getFullYear()} All rights reserved</p>
          </div>
        </div>
      </footer>

      {showHelp && (
        <div className="help-panel">
          <h2>Help</h2>
          <div className="help-content">
            <section>
              <h3>Getting Started</h3>
              <p>Welcome to the Hyperspectral Image Anomaly Detection application. Here's how to use it:</p>
              <ul>
                <li>Select a dataset from the dropdown menu</li>
                <li>Upload your hyperspectral image file</li>
                <li>Wait for the processing to complete</li>
                <li>View the results in the table below</li>
              </ul>
            </section>
            <section>
              <h3>Features</h3>
              <ul>
                <li>Dark/Light mode toggle</li>
                <li>Keyboard shortcuts for quick access</li>
                <li>Real-time backend status monitoring</li>
                <li>Toast notifications for important updates</li>
              </ul>
            </section>
          </div>
          <button className="close-btn" onClick={toggleHelp}>Close</button>
        </div>
      )}

      {showShortcuts && (
        <div className="shortcuts-panel">
          <h2>Keyboard Shortcuts</h2>
          <div className="shortcuts-content">
            <div className="shortcut-item">
              <kbd>Ctrl/Cmd</kbd> + <kbd>D</kbd>
              <span>Toggle Dark Mode</span>
            </div>
            <div className="shortcut-item">
              <kbd>Esc</kbd>
              <span>Close Toast Notifications</span>
            </div>
            <div className="shortcut-item">
              <kbd>Ctrl/Cmd</kbd> + <kbd>H</kbd>
              <span>Show/Hide Help</span>
            </div>
            <div className="shortcut-item">
              <kbd>Ctrl/Cmd</kbd> + <kbd>S</kbd>
              <span>Show/Hide Shortcuts</span>
            </div>
          </div>
          <button className="close-btn" onClick={toggleShortcuts}>Close</button>
        </div>
      )}
    </div>
  );
}

export default App;
