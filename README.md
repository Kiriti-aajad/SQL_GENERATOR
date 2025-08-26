┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT INTERFACES                        │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Frontend  │  FastAPI REST API  │  WebSocket API      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLIGENT ORCHESTRATOR                     │
├─────────────────────────────────────────────────────────────────┤
│  • Dynamic Intent Routing    • Request Coordination            │
│  • Hybrid Processing Logic   • Performance Optimization        │
│  • Context Management        • Error Recovery                  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  NLP PROCESSOR  │ │   SCHEMA    │ │   PROMPT    │ │     SQL     │
│                 │ │  SEARCHER   │ │   BUILDER   │ │  GENERATOR  │
├─────────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤
│ • Intent Class. │ │ • BM25 Eng. │ │ • Context   │ │ • DeepSeek  │
│ • Entity Extr.  │ │ • FAISS Eng.│ │ • Template  │ │ • Mathstral │
│ • Domain Map.   │ │ • ChromaDB  │ │ • Optimiz.  │ │ • Ensemble  │
│ • Query Enrich. │ │ • Semantic  │ │ • Validation│ │ • Correction│
└─────────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SQL EXECUTOR                               │
├─────────────────────────────────────────────────────────────────┤
│  • Syntax Validation    • Query Execution    • Result Format   │
│  • Performance Monitor  • Error Handling     • Security Check  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│              MONITORING & ANALYTICS LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  • Performance Tracking  • Usage Analytics  • Health Monitor   │
│  • Model Metrics         • Error Analytics  • System Alerts    │
└─────────────────────────────────────────────────────────────────┘
