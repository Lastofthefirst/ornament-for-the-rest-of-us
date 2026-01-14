import { createSignal, createEffect, For, Show, onMount, onCleanup } from 'solid-js';
import type { PageData, Issue } from './types';
import './App.css';

const API_BASE = '';

// SVG Icons as components
const ChevronLeft = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M15 18l-6-6 6-6" />
  </svg>
);

const ChevronRight = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M9 18l6-6-6-6" />
  </svg>
);

const Sparkles = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const ZoomIn = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <circle cx="11" cy="11" r="8" />
    <path d="M21 21l-4.35-4.35M11 8v6M8 11h6" />
  </svg>
);

const ZoomOut = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <circle cx="11" cy="11" r="8" />
    <path d="M21 21l-4.35-4.35M8 11h6" />
  </svg>
);

const Check = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M20 6L9 17l-5-5" />
  </svg>
);

const X = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M18 6L6 18M6 6l12 12" />
  </svg>
);

const Folder = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
  </svg>
);

function App() {
  // Folder state
  const [folderPath, setFolderPath] = createSignal('');
  const [currentFolder, setCurrentFolder] = createSignal<string | null>(null);
  const [folderError, setFolderError] = createSignal<string | null>(null);
  const [isLoadingFolder, setIsLoadingFolder] = createSignal(false);

  // Page state
  const [pages, setPages] = createSignal<PageData[]>([]);
  const [currentIndex, setCurrentIndex] = createSignal(0);
  const [isLoading, setIsLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [editedText, setEditedText] = createSignal('');
  const [hasChanges, setHasChanges] = createSignal(false);
  const [llmStatus, setLlmStatus] = createSignal<'idle' | 'analyzing' | 'error'>('idle');
  const [showSuggestions, setShowSuggestions] = createSignal(true);
  const [zoom, setZoom] = createSignal(100);

  const currentPage = () => pages()[currentIndex()];
  const pendingIssues = () => currentPage()?.issues?.filter(i => i.status === 'pending') ?? [];
  const progress = () => pages().length > 0 ? ((currentIndex() + 1) / pages().length) * 100 : 0;

  // Check status on mount
  onMount(async () => {
    try {
      const res = await fetch(`${API_BASE}/status`);
      if (res.ok) {
        const data = await res.json();
        if (data.ready && data.folder) {
          setCurrentFolder(data.folder);
          await loadPages();
        } else {
          setIsLoading(false);
        }
      } else {
        setIsLoading(false);
      }
    } catch (e) {
      setIsLoading(false);
    }
  });

  const loadPages = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/pages`);
      if (!res.ok) throw new Error('Failed to load pages');
      const data = await res.json();
      setPages(data.pages);
      setCurrentFolder(data.folder);
      if (data.pages.length > 0) {
        setEditedText(data.pages[0].text);
      }
      setIsLoading(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
      setIsLoading(false);
    }
  };

  const openFolder = async () => {
    const path = folderPath().trim();
    if (!path) return;

    setIsLoadingFolder(true);
    setFolderError(null);

    try {
      const res = await fetch(`${API_BASE}/folder`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Failed to open folder');
      }

      const data = await res.json();
      setCurrentFolder(data.folder);
      await loadPages();
    } catch (e) {
      setFolderError(e instanceof Error ? e.message : 'Failed to open folder');
    } finally {
      setIsLoadingFolder(false);
    }
  };

  const changeFolder = () => {
    setCurrentFolder(null);
    setPages([]);
    setFolderPath('');
    setIsLoading(false);
  };

  // Update edited text when page changes
  createEffect(() => {
    const page = currentPage();
    if (page) {
      setEditedText(page.text);
      setHasChanges(false);
    }
  });

  const handleTextChange = (newText: string) => {
    setEditedText(newText);
    setHasChanges(newText !== currentPage()?.text);
  };

  const savePage = async () => {
    const page = currentPage();
    if (!page || !hasChanges()) return;

    try {
      const res = await fetch(`${API_BASE}/pages/${page.pageNumber}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: editedText() }),
      });
      if (!res.ok) throw new Error('Failed to save');

      setPages(prev => prev.map((p, i) =>
        i === currentIndex() ? { ...p, text: editedText() } : p
      ));
      setHasChanges(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed');
    }
  };

  const analyzeWithLLM = async () => {
    const page = currentPage();
    if (!page || llmStatus() === 'analyzing') return;

    setLlmStatus('analyzing');
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pageNumber: page.pageNumber,
          text: editedText()
        }),
      });
      if (!res.ok) throw new Error('Analysis failed');

      const data = await res.json();
      setPages(prev => prev.map((p, i) =>
        i === currentIndex() ? { ...p, issues: data.issues, hasIssues: data.issues.length > 0 } : p
      ));
      setLlmStatus('idle');
      if (data.issues.length > 0) {
        setShowSuggestions(true);
      }
    } catch (e) {
      setLlmStatus('error');
      setTimeout(() => setLlmStatus('idle'), 2000);
    }
  };

  const applyFix = (issue: Issue) => {
    if (!issue.suggestion) return;
    const text = editedText();
    const newText = text.slice(0, issue.startOffset) + issue.suggestion + text.slice(issue.endOffset);
    setEditedText(newText);
    setHasChanges(true);

    setPages(prev => prev.map((p, i) =>
      i === currentIndex()
        ? { ...p, issues: p.issues.map(iss => iss.id === issue.id ? { ...iss, status: 'accepted' as const } : iss) }
        : p
    ));
  };

  const rejectFix = (issue: Issue) => {
    setPages(prev => prev.map((p, i) =>
      i === currentIndex()
        ? { ...p, issues: p.issues.map(iss => iss.id === issue.id ? { ...iss, status: 'rejected' as const } : iss) }
        : p
    ));
  };

  const goToPage = (index: number) => {
    if (index >= 0 && index < pages().length) {
      setCurrentIndex(index);
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    // Don't intercept when typing in inputs
    if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) {
      if (!e.altKey && !e.ctrlKey && !e.metaKey) return;
    }

    if (e.key === 'ArrowLeft' && e.altKey) {
      e.preventDefault();
      goToPage(currentIndex() - 1);
    } else if (e.key === 'ArrowRight' && e.altKey) {
      e.preventDefault();
      goToPage(currentIndex() + 1);
    } else if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      savePage();
    } else if (e.key === 'a' && (e.ctrlKey || e.metaKey) && e.shiftKey) {
      e.preventDefault();
      analyzeWithLLM();
    }
  };

  onMount(() => {
    window.addEventListener('keydown', handleKeyDown);
  });

  onCleanup(() => {
    window.removeEventListener('keydown', handleKeyDown);
  });

  const adjustZoom = (delta: number) => {
    setZoom(prev => Math.min(200, Math.max(50, prev + delta)));
  };

  // Folder Selection Screen
  if (!currentFolder() && !isLoading()) {
    return (
      <div class="app">
        <div class="folder-select">
          <div class="folder-select-content">
            <div class="folder-icon">
              <Folder />
            </div>
            <h1>OCR Review</h1>
            <p>Enter the path to your OCR output folder</p>

            <div class="folder-input-group">
              <input
                type="text"
                class="folder-input"
                placeholder="/path/to/ocr/output"
                value={folderPath()}
                onInput={(e) => setFolderPath(e.currentTarget.value)}
                onKeyDown={(e) => e.key === 'Enter' && openFolder()}
                autofocus
              />
              <button
                class="btn btn-primary"
                onClick={openFolder}
                disabled={isLoadingFolder() || !folderPath().trim()}
              >
                {isLoadingFolder() ? 'Loading...' : 'Open'}
              </button>
            </div>

            <Show when={folderError()}>
              <div class="folder-error">{folderError()}</div>
            </Show>

            <div class="folder-hint">
              <p>The folder should contain OCR output files like:</p>
              <code>page_0001.md, page_0001.json, page_0001.jpg</code>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div class="app">
      {/* Header */}
      <header class="header">
        <div class="header-left">
          <h1 class="title">OCR Review</h1>
          <Show when={currentPage()}>
            <span class="page-indicator">
              {currentIndex() + 1} / {pages().length}
            </span>
          </Show>
          <Show when={pages().length > 0}>
            <div class="progress-bar">
              <div class="progress-fill" style={{ width: `${progress()}%` }} />
            </div>
          </Show>
        </div>
        <div class="header-actions">
          <button class="btn btn-ghost" onClick={changeFolder} title="Change folder">
            <Folder />
          </button>
          <button
            class="btn btn-secondary"
            onClick={analyzeWithLLM}
            disabled={llmStatus() === 'analyzing' || !currentPage()}
          >
            <Sparkles />
            {llmStatus() === 'analyzing' ? 'Analyzing...' : llmStatus() === 'error' ? 'Error' : 'Analyze'}
          </button>
          <button
            class="btn btn-primary"
            onClick={savePage}
            disabled={!hasChanges()}
          >
            Save
          </button>
        </div>
      </header>

      {/* Error Banner */}
      <Show when={error()}>
        <div class="error-banner">
          {error()}
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      </Show>

      {/* Loading State */}
      <Show when={isLoading()}>
        <div class="loading">
          <div class="loading-spinner" />
          <span>Loading pages...</span>
        </div>
      </Show>

      {/* Main Content */}
      <Show when={!isLoading() && pages().length > 0}>
        <main class={`main ${showSuggestions() && pendingIssues().length > 0 ? 'with-suggestions' : ''}`}>
          {/* Image Panel */}
          <section class="panel image-panel">
            <div class="panel-header">
              <span class="panel-title">Source</span>
              <div class="panel-controls">
                <div class="zoom-controls">
                  <button class="btn-icon" onClick={() => adjustZoom(-10)} title="Zoom out">
                    <ZoomOut />
                  </button>
                  <span class="zoom-level">{zoom()}%</span>
                  <button class="btn-icon" onClick={() => adjustZoom(10)} title="Zoom in">
                    <ZoomIn />
                  </button>
                </div>
              </div>
            </div>
            <div class="image-container">
              <img
                src={`${API_BASE}/image/${currentPage()?.pageNumber}`}
                alt={`Page ${currentIndex() + 1}`}
                class="page-image"
                style={{ transform: `scale(${zoom() / 100})`, 'transform-origin': 'top center' }}
              />
            </div>
          </section>

          {/* Editor Panel */}
          <section class="panel editor-panel">
            <div class="panel-header">
              <span class="panel-title">Text</span>
              <Show when={hasChanges()}>
                <span class="status-badge unsaved">Unsaved</span>
              </Show>
            </div>
            <div class="text-editor-wrapper">
              <textarea
                class="text-editor"
                value={editedText()}
                onInput={(e) => handleTextChange(e.currentTarget.value)}
                spellcheck={false}
              />
            </div>
          </section>

          {/* Suggestions Panel */}
          <Show when={showSuggestions() && pendingIssues().length > 0}>
            <aside class="panel suggestions-panel">
              <div class="panel-header">
                <div style={{ display: 'flex', 'align-items': 'center', gap: '8px' }}>
                  <span class="panel-title">Suggestions</span>
                  <span class="suggestion-count">{pendingIssues().length}</span>
                </div>
                <button class="btn-icon" onClick={() => setShowSuggestions(false)} title="Close">
                  <X />
                </button>
              </div>
              <div class="suggestions-list">
                <For each={pendingIssues()}>
                  {(issue) => (
                    <div class={`suggestion-card severity-${issue.severity}`}>
                      <div class="suggestion-header">
                        <div class="suggestion-type">
                          {issue.type.replace(/_/g, ' ')}
                        </div>
                        <p class="suggestion-desc">{issue.description}</p>
                      </div>
                      <Show when={issue.suggestion}>
                        <div class="suggestion-diff">
                          <div class="diff-line removed">
                            {editedText().slice(issue.startOffset, issue.endOffset)}
                          </div>
                          <div class="diff-line added">
                            {issue.suggestion}
                          </div>
                        </div>
                        <div class="suggestion-actions">
                          <button class="btn btn-sm btn-success" onClick={() => applyFix(issue)}>
                            <Check /> Accept
                          </button>
                          <button class="btn btn-sm btn-secondary" onClick={() => rejectFix(issue)}>
                            <X /> Reject
                          </button>
                        </div>
                      </Show>
                    </div>
                  )}
                </For>
              </div>
            </aside>
          </Show>
        </main>

        {/* Page Navigation */}
        <nav class="page-nav">
          <button
            class="nav-btn"
            onClick={() => goToPage(currentIndex() - 1)}
            disabled={currentIndex() === 0}
            title="Previous page (Alt+Left)"
          >
            <ChevronLeft />
          </button>

          <div class="page-strip">
            <For each={pages()}>
              {(page, index) => (
                <button
                  class={`page-thumb ${index() === currentIndex() ? 'active' : ''} ${page.hasIssues ? 'has-issues' : ''}`}
                  onClick={() => goToPage(index())}
                  title={`Page ${index() + 1}${page.hasIssues ? ' (has issues)' : ''}`}
                >
                  {index() + 1}
                </button>
              )}
            </For>
          </div>

          <button
            class="nav-btn"
            onClick={() => goToPage(currentIndex() + 1)}
            disabled={currentIndex() === pages().length - 1}
            title="Next page (Alt+Right)"
          >
            <ChevronRight />
          </button>
        </nav>
      </Show>

      {/* Empty State - folder loaded but no pages */}
      <Show when={!isLoading() && currentFolder() && pages().length === 0}>
        <div class="empty-state">
          <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <h2>No pages found</h2>
          <p>No OCR page files found in the selected folder.</p>
          <button class="btn btn-secondary" onClick={changeFolder}>
            Choose Different Folder
          </button>
        </div>
      </Show>
    </div>
  );
}

export default App;
