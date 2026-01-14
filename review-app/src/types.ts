/** OCR block from dots.ocr JSON output */
export interface OCRBlock {
  bbox: [number, number, number, number];
  category: 'Text' | 'Section-header' | 'Title' | 'Caption' | 'Footnote' | 'List-item';
  text: string;
}

/** Page data loaded from the OCR output */
export interface PageData {
  pageNumber: number;
  imagePath: string;
  markdownPath: string;
  jsonPath: string;
  blocks: OCRBlock[];
  text: string;
  hasIssues: boolean;
  issues: Issue[];
}

/** Issue detected by LLM or heuristics */
export interface Issue {
  id: string;
  type: IssueType;
  severity: 'low' | 'medium' | 'high';
  description: string;
  startOffset: number;
  endOffset: number;
  suggestion?: string;
  status: 'pending' | 'accepted' | 'rejected' | 'edited';
}

export type IssueType =
  | 'garbled_text'      // Nonsense characters
  | 'ocr_confusion'     // 0/O, 1/l/I confusion
  | 'missing_space'     // Words run together
  | 'extra_space'       // Spurious spaces
  | 'broken_word'       // Word split across lines
  | 'header_issue'      // Running header problems
  | 'formatting'        // Markdown formatting issues
  | 'spelling'          // Potential misspelling
  | 'repetition'        // Repeated text (hallucination)
  | 'truncation';       // Text cut off

/** LLM tool call for suggesting fixes */
export interface SuggestionToolCall {
  tool: 'suggest_fix';
  arguments: {
    issue_type: IssueType;
    description: string;
    original_text: string;
    suggested_text: string;
    start_offset: number;
    end_offset: number;
    confidence: number;
  };
}

/** Book/project manifest */
export interface BookManifest {
  title: string;
  sourcePath: string;
  pageCount: number;
  createdAt: string;
  lastModified: string;
}

/** App state */
export interface AppState {
  manifest: BookManifest | null;
  pages: PageData[];
  currentPageIndex: number;
  isLoading: boolean;
  error: string | null;
  llmStatus: 'idle' | 'analyzing' | 'error';
}
