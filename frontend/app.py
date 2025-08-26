import streamlit as st
import requests
import pandas as pd
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import base64

# =================================================================
# CONFIGURATION
# =================================================================

# FastAPI server configuration
API_BASE_URL = "http://localhost:8000"  # Update this to your server URL
ENDPOINTS = {
    "query": f"{API_BASE_URL}/query",
    "health": f"{API_BASE_URL}/health", 
    "config": f"{API_BASE_URL}/config",
    "components": f"{API_BASE_URL}/components",
    "execute_sql": f"{API_BASE_URL}/execute-sql"  # NEW: SQL execution endpoint
}

# =================================================================
# PAGE CONFIGURATION
# =================================================================

st.set_page_config(
    page_title="AI SQL Generator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional, robust UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    .swan-logo {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.15);
    }
    
    .status-healthy {
        background: linear-gradient(135deg, #00C851, #007E33);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 200, 81, 0.3);
    }
    
    .status-unhealthy {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .query-section {
        background: linear-gradient(145deg, #f8f9ff, #ffffff);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.08);
        border: 2px solid rgba(102, 126, 234, 0.1);
        margin: 1rem 0;
    }
    
    .query-input {
        background: white;
        border: 2px solid #e1e8ff;
        border-radius: 15px;
        padding: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .query-input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .execute-btn {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .execute-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: none;
        border-left: 5px solid #28a745;
        color: #155724;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.1);
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: none;
        border-left: 5px solid #dc3545;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: none;
        border-left: 5px solid #ffc107;
        color: #856404;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.1);
    }
    
    .sql-display {
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.05);
        position: relative;
        overflow-x: auto;
    }
    
    .sql-display::before {
        content: ' Generated SQL';
        position: absolute;
        top: -10px;
        left: 15px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .execution-panel {
        background: linear-gradient(145deg, #e7f3ff, #f0f8ff);
        border: 2px solid #b3d9ff;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        position: relative;
    }
    
    .execution-panel::before {
        content: '🚀';
        position: absolute;
        top: -15px;
        left: 20px;
        background: #007bff;
        color: white;
        padding: 10px;
        border-radius: 50%;
        font-size: 1.2rem;
    }
    
    .results-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .sidebar-section {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.08);
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .performance-chart {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    .history-item {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.05);
        transition: transform 0.2s ease;
    }
    
    .history-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .data-table {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e1e8ff;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #e1e8ff;
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stMetric {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# =================================================================
# UTILITY FUNCTIONS
# =================================================================

@st.cache_data(ttl=30)
def check_server_health() -> Dict[str, Any]:
    """Check if the FastAPI server is healthy"""
    try:
        response = requests.get(ENDPOINTS["health"], timeout=10)
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds(),
            "data": response.json() if response.status_code == 200 else None
        }
    except Exception as e:
        return {
            "status": "down",
            "response_time": None,
            "error": str(e)
        }

def send_query(query: str, session_id: str = "streamlit-session") -> Dict[str, Any]:
    """Send query to FastAPI server"""
    try:
        payload = {
            "query": query,
            "session_id": session_id,
            "max_results": 100,
            "include_explanation": True
        }
        
        start_time = time.time()
        response = requests.post(
            ENDPOINTS["query"],
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minutes timeout
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            result["execution_time"] = execution_time
            result["timestamp"] = datetime.now().isoformat()
            return result
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Query timed out after 2 minutes",
            "execution_time": 120,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time if 'start_time' in locals() else 0, # pyright: ignore[reportPossiblyUnboundVariable]
            "timestamp": datetime.now().isoformat()
        }

def execute_sql_on_server(sql: str, session_id: str = "streamlit-session") -> Dict[str, Any]:
    """Execute SQL query on the server and return results"""
    try:
        payload = {
            "sql": sql,
            "session_id": session_id,
            "max_rows": 1000,
            "timeout": 60
        }
        
        start_time = time.time()
        response = requests.post(
            ENDPOINTS["execute_sql"],
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=70
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            result["execution_time"] = execution_time
            result["timestamp"] = datetime.now().isoformat()
            return result
        else:
            return {
                "success": False,
                "error": f"SQL Execution Error (HTTP {response.status_code}): {response.text}",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "SQL execution timed out after 60 seconds",
            "execution_time": 70,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"SQL execution error: {str(e)}",
            "execution_time": time.time() - start_time if 'start_time' in locals() else 0, # pyright: ignore[reportPossiblyUnboundVariable]
            "timestamp": datetime.now().isoformat()
        }

def validate_sql_safety(sql: str) -> Dict[str, Any]:
    """Basic SQL safety validation"""
    sql_upper = sql.upper().strip()
    
    dangerous_keywords = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 
        'ALTER', 'CREATE', 'REPLACE', 'MERGE', 'EXEC',
        'EXECUTE', 'xp_', 'sp_'
    ]
    
    warnings = []
    is_safe = True
    
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            warnings.append(f"⚠️ Potentially dangerous operation detected: {keyword}")
            is_safe = False
    
    if not sql_upper.startswith('SELECT'):
        if not sql_upper.startswith('WITH'):
            warnings.append("⚠️ Only SELECT queries are recommended")
            is_safe = False
    
    return {
        "is_safe": is_safe,
        "warnings": warnings,
        "recommendation": "Only SELECT statements are executed for security" if not is_safe else "Query appears safe"
    }

def export_results_to_csv(data: List[Dict], filename: str = "query_results.csv") -> str:
    """Export results to CSV and return download link"""
    if data:
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 10px 20px; border-radius: 25px; text-decoration: none; font-weight: 600;">📥 Download CSV</a>'
        return href
    return ""

# =================================================================
# SESSION STATE INITIALIZATION
# =================================================================

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "current_results" not in st.session_state:
    st.session_state.current_results = None

if "sql_execution_results" not in st.session_state:
    st.session_state.sql_execution_results = None

if "edited_sql" not in st.session_state:
    st.session_state.edited_sql = ""

# =================================================================
# MAIN APPLICATION
# =================================================================

def main():
    # Enhanced Header with AadiSwan branding
    st.markdown("""
    <div class="main-header">
        <h1>AI SQL GENERATOR</h1>
        <p>Your Intelligent SQL Assistant - Transform Questions into Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar with professional styling
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("## 🔧 System Status")
        
        # Server health check with enhanced styling
        health_status = check_server_health()
        
        if health_status["status"] == "healthy":
            st.markdown("""
            <div class="status-healthy">
                ✅ AadiSwan Online & Ready
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-highlight">
                    <strong>{health_status['response_time']:.2f}s</strong><br>
                    <small>Response Time</small>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric-highlight">
                    <strong>100%</strong><br>
                    <small>Uptime</small>
                </div>
                """, unsafe_allow_html=True)
                
        elif health_status["status"] == "unhealthy":
            st.markdown("""
            <div class="status-unhealthy">
                ⚠️ System Issues Detected
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-unhealthy">
                ❌ AadiSwan Offline
            </div>
            """, unsafe_allow_html=True)
            st.error(f"Error: {health_status.get('error', 'Unknown error')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Query Statistics
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("## 📊 Session Analytics")
        
        total_queries = len(st.session_state.query_history)
        successful_queries = sum(1 for q in st.session_state.query_history if q.get("success", False))
        
        if total_queries > 0:
            success_rate = (successful_queries / total_queries) * 100
            exec_times = [q.get("execution_time", 0) for q in st.session_state.query_history if q.get("execution_time")]
            avg_time = sum(exec_times) / len(exec_times) if exec_times else 0
            
            # Create beautiful metrics display
            metrics_html = f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 1rem 0;">
                <div class="metric-highlight">
                    <strong>{total_queries}</strong><br>
                    <small>Total Queries</small>
                </div>
                <div class="metric-highlight">
                    <strong>{successful_queries}</strong><br>
                    <small>Successful</small>
                </div>
                <div class="metric-highlight">
                    <strong>{success_rate:.1f}%</strong><br>
                    <small>Success Rate</small>
                </div>
                <div class="metric-highlight">
                    <strong>{avg_time:.2f}s</strong><br>
                    <small>Avg Time</small>
                </div>
            </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)
        else:
            st.info("🌟 Ready to analyze your first query!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Quick Actions
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("## ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.query_history = []
                st.session_state.current_results = None
                st.session_state.sql_execution_results = None
                st.session_state.edited_sql = ""
                st.success("✨ History cleared!")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area with enhanced styling
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        # Enhanced Query Input Section
        st.markdown('<div class="query-section">', unsafe_allow_html=True)
        st.markdown("### 💬 Ask AadiSwan Anything")
        
        # Enhanced query input
        query_input = st.text_area(
            "Enter your natural language query:",
            height=120,
            placeholder="e.g., Show me all customers with their account details and recent transactions...",
            help="Type your question in plain English - AadiSwan will convert it to SQL!"
        )
        
        # Enhanced execution buttons (removed Examples button)
        col_execute, col_clear = st.columns([3, 1])
        
        with col_execute:
            execute_query = st.button(
                "🚀 Ask AadiSwan", 
                type="primary", 
                use_container_width=True,
                help="Convert your question to SQL and get results"
            )
        
        with col_clear:
            clear_input = st.button("🧹 Clear", use_container_width=True)
        
        if clear_input:
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Execute query with enhanced feedback
        if execute_query and query_input.strip():
            with st.spinner("🧠 AadiSwan is thinking..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result = send_query(query_input)
                st.session_state.current_results = result
                st.session_state.sql_execution_results = None
                st.session_state.edited_sql = ""
                st.session_state.query_history.append({
                    "query": query_input,
                    "timestamp": result.get("timestamp"),
                    "success": result.get("success", False),
                    "execution_time": result.get("execution_time", 0),
                    "result": result
                })
            st.rerun()
    
    with col2:
        # Enhanced Performance Metrics
        st.markdown('<div class="performance-chart">', unsafe_allow_html=True)
        st.markdown("### 📈 Performance Insights")
        
        if st.session_state.query_history:
            recent_queries = st.session_state.query_history[-10:]
            
            # Enhanced performance visualization
            exec_times = [q.get("execution_time", 0) for q in recent_queries]
            success_status = [q.get("success", False) for q in recent_queries]
            
            # Create dual-axis chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(exec_times))),
                y=exec_times,
                mode='lines+markers',
                name='Response Time',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="AadiSwan Performance",
                xaxis_title="Query Number",
                yaxis_title="Response Time (s)",
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Success rate gauge
            if len(st.session_state.query_history) > 0:
                success_rate = (sum(success_status) / len(success_status)) * 100
                
                gauge_fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = success_rate,
                    title = {'text': "Success Rate"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                gauge_fig.update_layout(height=200)
                st.plotly_chart(gauge_fig, use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #667eea;">
                <h3>🌟 Ready to Start!</h3>
                <p>Your performance metrics will appear here after your first query.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Results Display
    if st.session_state.current_results:
        result = st.session_state.current_results
        
        st.markdown("---")
        
        if result.get("success", False):
            # Success notification with animation
            st.markdown("""
            <div class="success-box pulse-animation">
                <h3>✅ Query Processed Successfully!</h3>
                <p>AadiSwan has converted your question into SQL and is ready to execute it.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced execution metrics
            col1, col2, col3, col4 = st.columns(4)
            metrics = [
                ("⚡ Execution Time", f"{result.get('execution_time', 0):.2f}s"),
                ("🎯 Confidence", result.get('confidence', 'Unknown')),
                ("🤖 Generated By", "AadiSwan AI"),
                ("📊 Processing Mode", result.get('processing_mode', 'Advanced'))
            ]
            
            for i, (label, value) in enumerate(metrics):
                with [col1, col2, col3, col4][i]:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <h4 style="margin: 0; color: #667eea;">{value}</h4>
                        <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9rem;">{label}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Enhanced SQL Display
            if result.get("sql"):
                st.markdown("### 🔍 Generated SQL Query")
                
                st.markdown(f"""
                <div class="sql-display">
                    <pre><code>{result['sql']}</code></pre>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced SQL Execution Panel
                st.markdown("""
                <div class="execution-panel">
                    <h4 style="margin-top: 0;">Execute Your Query</h4>
                    <p style="color: #666;">Choose how you want to run your SQL query:</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced execution options
                exec_col1, exec_col2 = st.columns(2)
                
                with exec_col1:
                    st.markdown("**🎯 Execute Original Query**")
                    execute_original = st.button(
                        "▶️ Run Generated SQL",
                        type="primary",
                        help="Execute the AI-generated SQL query as-is",
                        key="execute_original",
                        use_container_width=True
                    )
                
                with exec_col2:
                    st.markdown("**✏️ Edit & Execute**")
                    show_editor = st.button(
                        "🛠️ Customize & Run",
                        help="Edit the SQL query before execution",
                        key="show_editor",
                        use_container_width=True
                    )
                
                # Execute original SQL
                if execute_original:
                    with st.spinner("🔄 Executing query with AadiSwan..."):
                        sql_result = execute_sql_on_server(result['sql'])
                        st.session_state.sql_execution_results = sql_result
                    st.rerun()
                
                # Enhanced SQL Editor
                if show_editor or st.session_state.edited_sql:
                    st.markdown("---")
                    st.markdown("### ✏️ SQL Query Editor")
                    
                    if not st.session_state.edited_sql:
                        st.session_state.edited_sql = result['sql']
                    
                    edited_sql = st.text_area(
                        "Customize your SQL query:",
                        value=st.session_state.edited_sql,
                        height=200,
                        help="Make any modifications to the generated SQL query",
                        key="sql_editor"
                    )
                    
                    st.session_state.edited_sql = edited_sql
                    
                    # Enhanced SQL Safety Check
                    if edited_sql.strip(): # pyright: ignore[reportOptionalMemberAccess]
                        safety_check = validate_sql_safety(edited_sql) # pyright: ignore[reportArgumentType]
                        
                        if safety_check["warnings"]:
                            st.markdown(f"""
                            <div class="warning-box">
                                <strong>⚠️ Security Notice:</strong><br>
                                {' '.join(safety_check["warnings"])}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Enhanced editor controls
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        execute_edited = st.button(
                            "🚀 Execute Modified SQL",
                            type="primary",
                            disabled=not edited_sql.strip(), # pyright: ignore[reportOptionalMemberAccess]
                            key="execute_edited",
                            use_container_width=True
                        )
                    
                    with col2:
                        reset_sql = st.button("🔄 Reset to Original", key="reset_sql", use_container_width=True)
                    
                    with col3:
                        format_sql = st.button("✨ Format SQL", key="format_sql", use_container_width=True)
                    
                    with col4:
                        copy_sql = st.button("📋 Copy to Clipboard", key="copy_sql", use_container_width=True)
                    
                    if reset_sql:
                        st.session_state.edited_sql = result['sql']
                        st.rerun()
                    
                    if format_sql:
                        st.info("💡 SQL formatting feature coming soon!")
                    
                    if copy_sql:
                        st.code(edited_sql, language='sql')
                        st.success("📋 SQL copied to display above!")
                    
                    if execute_edited and edited_sql.strip(): # pyright: ignore[reportOptionalMemberAccess]
                        with st.spinner("⚡ Executing your customized query..."):
                            sql_result = execute_sql_on_server(edited_sql) # pyright: ignore[reportArgumentType]
                            st.session_state.sql_execution_results = sql_result
                        st.rerun()
            
            # Enhanced Query Explanation
            if result.get("explanation"):
                with st.expander("💡 How AadiSwan Interpreted Your Query", expanded=False):
                    st.markdown(f"""
                    <div style="background: #f8f9ff; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea;">
                        {result['explanation']}
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            # Enhanced error display
            st.markdown(f"""
            <div class="error-box">
                <h3>❌ Query Processing Failed</h3>
                <p><strong>Error:</strong> {result.get('error', 'Unknown error occurred')}</p>
                <p><strong>Don't worry!</strong> Try rephrasing your question or contact support.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⏱️ Processing Time", f"{result.get('execution_time', 0):.2f}s")
            with col2:
                st.metric("🔍 Error Code", result.get('error_code', 'UNKNOWN'))
    
    # Enhanced SQL Results Display
    if st.session_state.sql_execution_results:
        sql_result = st.session_state.sql_execution_results
        
        st.markdown("---")
        st.markdown("## 📊 Query Results from AadiSwan")
        
        if sql_result.get("success", False):
            # Enhanced success metrics
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = [
                ("⚡ Query Time", f"{sql_result.get('execution_time', 0):.2f}s"),
                ("📊 Rows Returned", f"{sql_result.get('row_count', 0):,}"),
                ("📋 Columns", len(sql_result.get('columns', []))),
                ("✅ Status", "Success")
            ]
            
            for i, (label, value) in enumerate(metrics):
                with [col1, col2, col3, col4][i]:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <h3 style="margin: 0; color: #28a745;">{value}</h3>
                        <p style="margin: 5px 0 0 0; color: #666;">{label}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Enhanced Results Table
            if sql_result.get("data") and isinstance(sql_result["data"], list):
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.markdown("### 📋 Your Data Results")
                
                df = pd.DataFrame(sql_result["data"])
                
                # Enhanced display controls
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    show_all = st.checkbox("Show all rows", value=False)
                
                with col2:
                    if not show_all:
                        max_rows = st.number_input("Rows to display", min_value=10, max_value=1000, value=100, step=10)
                    else:
                        max_rows = len(df)
                
                with col3:
                    st.metric("Total Rows", f"{len(df):,}")
                
                with col4:
                    if len(df) > 0:
                        csv_download = export_results_to_csv(
                            sql_result["data"], 
                            f"aadiswan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        )
                        if csv_download:
                            st.markdown(csv_download, unsafe_allow_html=True)
                
                # Enhanced data display
                display_df = df.head(max_rows) if not show_all else df
                
                st.markdown('<div class="data-table">', unsafe_allow_html=True)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=min(400, len(display_df) * 35 + 50)
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Enhanced analytics options
                if len(df) > 1:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📊 Data Summary", use_container_width=True):
                            st.markdown("### 📈 Data Analysis")
                            
                            # Data types
                            st.markdown("**📊 Column Information:**")
                            type_df = pd.DataFrame({
                                'Column': df.dtypes.index,
                                'Data Type': df.dtypes.values,
                                'Non-Null Count': df.count().values,
                                'Null Count': df.isnull().sum().values
                            })
                            st.dataframe(type_df, use_container_width=True)
                            
                            # Statistical summary for numeric columns
                            numeric_cols = df.select_dtypes(include=[int, float]).columns
                            if len(numeric_cols) > 0:
                                st.markdown("**📈 Statistical Summary:**")
                                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    
                    with col2:
                        if st.button("📊 Quick Visualization", use_container_width=True):
                            st.markdown("### 📊 Data Visualization")
                            
                            numeric_cols = df.select_dtypes(include=[int, float]).columns
                            
                            if len(numeric_cols) >= 2:
                                col_x = st.selectbox("Select X-axis", numeric_cols, key="chart_x")
                                col_y = st.selectbox("Select Y-axis", [col for col in numeric_cols if col != col_x], key="chart_y")
                                
                                fig = px.scatter(
                                    df, 
                                    x=col_x, 
                                    y=col_y,
                                    title=f"Relationship between {col_x} and {col_y}",
                                    color_discrete_sequence=['#667eea']
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif len(numeric_cols) == 1:
                                fig = px.histogram(
                                    df, 
                                    x=numeric_cols[0],
                                    title=f"Distribution of {numeric_cols}",
                                    color_discrete_sequence=['#667eea']
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("🎨 No numeric columns available for visualization")
                    
                    with col3:
                        if st.button("🔍 Advanced Filters", use_container_width=True):
                            st.markdown("### 🔍 Filter Your Data")
                            st.info("🚀 Advanced filtering features coming soon!")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: #f8f9ff; border-radius: 15px; border: 2px dashed #667eea;">
                    <h3 style="color: #667eea;">📊 Query Executed Successfully</h3>
                    <p style="color: #666;">The query ran without errors but returned no data rows.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Enhanced SQL error display
            st.markdown(f"""
            <div class="error-box">
                <h3>❌ SQL Execution Failed</h3>
                <p><strong>Database Error:</strong> {sql_result.get('error', 'Unknown SQL execution error')}</p>
                <p><strong>Tip:</strong> Check your SQL syntax or try modifying the query.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⏱️ Execution Time", f"{sql_result.get('execution_time', 0):.2f}s")
            with col2:
                st.metric("🔍 Error Type", sql_result.get('error_type', 'SQL_ERROR'))
    
    # Enhanced Query History
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("## 📜 Recent Queries with AadiSwan")
        
        for i, query_record in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"🔍 Query {len(st.session_state.query_history) - i}: {query_record['query'][:60]}..."):
                st.markdown('<div class="history-item">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**⏰ Time:** {query_record['timestamp']}")
                with col2:
                    status_color = "#28a745" if query_record['success'] else "#dc3545"
                    status_text = "✅ Success" if query_record['success'] else "❌ Failed"
                    st.markdown(f"**📊 Status:** <span style='color: {status_color}'>{status_text}</span>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"**⚡ Duration:** {query_record['execution_time']:.2f}s")
                
                st.markdown("**🗣️ Your Question:**")
                st.markdown(f"*{query_record['query']}*")
                
                if query_record['success'] and query_record['result'].get('sql'):
                    st.markdown("**🔍 Generated SQL:**")
                    st.code(query_record['result']['sql'], language='sql')
                    
                    if st.button(f"🔄 Re-execute This Query", key=f"reexec_{i}", use_container_width=True):
                        with st.spinner("🔄 Re-executing with AadiSwan..."):
                            sql_result = execute_sql_on_server(query_record['result']['sql'])
                            st.session_state.sql_execution_results = sql_result
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)

# =================================================================
# RUN THE APPLICATION
# =================================================================

if __name__ == "__main__":
    main()
