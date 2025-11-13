import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from openai import OpenAI

st.set_page_config(
    page_title="Health Score - 0~6 months Churn Insights",
    page_icon="📊",
    layout="wide"
)

st.title("🎯 Single Location Customer Health Score - 0~6 months Churn Insights")

# Check if model outputs exist
model_outputs_path = Path("model_outputs")
if not model_outputs_path.exists():
    st.error("Model outputs not found. Please run the regression model first.")
    st.stop()

# Load model results
@st.cache_data
def load_model_data():
    """Load all model output files"""
    try:
        # Load coefficient data
        coefficients = pd.read_csv("model_outputs/logit_coefficients.csv")
        
        # Load scored dataset
        scored_data = pd.read_csv("model_outputs/scored_dataset_with_risk.csv")
        
        # Load all splits data if exists
        all_splits_path = "model_outputs/scored_rows_all_splits.csv"
        if os.path.exists(all_splits_path):
            all_splits = pd.read_csv(all_splits_path)
        else:
            all_splits = None
            
        return coefficients, scored_data, all_splits
    except Exception as e:
        st.error(f"Error loading model data: {str(e)}")
        return None, None, None

# Load data
coefficients, scored_data, all_splits = load_model_data()

if coefficients is None or scored_data is None:
    st.stop()

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["Model Overview", "Feature Importance", "Risk Analysis", "Customer Segments", "Churn Segmentation", "AI Assistant", "Detailed Results"]
)

if page == "Model Overview":
    st.header("📈 Model Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate key metrics
    total_customers = len(scored_data)
    churned_customers = len(scored_data[scored_data['y_true'] == 1])
    retained_customers = len(scored_data[scored_data['y_true'] == 0])
    churn_rate = (churned_customers / total_customers) * 100
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("Churned", f"{churned_customers:,}")
    with col3:
        st.metric("Retained", f"{retained_customers:,}")
    with col4:
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    # Model accuracy metrics
    if 'y_pred' in scored_data.columns:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(scored_data['y_true'], scored_data['y_pred'])
        precision = precision_score(scored_data['y_true'], scored_data['y_pred'])
        recall = recall_score(scored_data['y_true'], scored_data['y_pred'])
        f1 = f1_score(scored_data['y_true'], scored_data['y_pred'])
        
        st.subheader("🎯 Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            st.metric("Recall", f"{recall:.3f}")
        with col4:
            st.metric("F1-Score", f"{f1:.3f}")
    
    # Distribution of churn probability
    st.subheader("📊 Churn Probability Distribution")
    fig = px.histogram(
        scored_data, 
        x='p_churn', 
        color='Churn or Retain',
        nbins=50,
        title="Distribution of Churn Probabilities",
        labels={'p_churn': 'Churn Probability', 'count': 'Number of Customers'}
    )
    st.plotly_chart(fig, width='stretch')

elif page == "Feature Importance":
    st.header("🔍 Feature Importance Analysis")
    
    # Sort coefficients by absolute value
    coef_viz = coefficients.copy()
    coef_viz['abs_coefficient'] = abs(coef_viz['coefficient'])
    coef_viz = coef_viz.sort_values('abs_coefficient', ascending=True)
    
    # Create feature importance plot
    fig = go.Figure()
    
    colors = ['red' if x < 0 else 'green' for x in coef_viz['coefficient']]
    
    fig.add_trace(go.Bar(
        y=coef_viz['variable'],
        x=coef_viz['coefficient'],
        orientation='h',
        marker_color=colors,
        text=coef_viz['coefficient'].round(3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Importance (Logistic Regression Coefficients)",
        xaxis_title="Coefficient Value",
        yaxis_title="Features",
        height=max(400, len(coef_viz) * 25)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show coefficient details
    st.subheader("📋 Coefficient Details")
    
    # Add interpretation column
    coef_display = coefficients.copy()
    coef_display['interpretation'] = coef_display['coefficient'].apply(
        lambda x: "Increases churn risk" if x > 0 else "Decreases churn risk"
    )
    
    st.dataframe(
        coef_display[['variable', 'coefficient', 'odds_ratio', 'p_value', 'interpretation']].round(4),
        width='stretch'
    )

elif page == "Risk Analysis":
    st.header("⚠️ Customer Risk Analysis")
    
    # Risk distribution
    st.subheader("📊 Business Risk Index Distribution")
    
    fig = px.histogram(
        scored_data,
        x='business_risk_index',
        nbins=50,
        title="Distribution of Business Risk Index",
        labels={'business_risk_index': 'Business Risk Index', 'count': 'Number of Customers'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # High-risk customers
    st.subheader("🚨 High-Risk Customers")
    
    # Define risk thresholds
    high_risk_threshold = scored_data['business_risk_index'].quantile(0.8)
    high_risk_customers = scored_data[scored_data['business_risk_index'] >= high_risk_threshold]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("High-Risk Customers", len(high_risk_customers))
    with col2:
        st.metric("Risk Threshold", f"{high_risk_threshold:.3f}")
    
    # Show high-risk customer details
    if len(high_risk_customers) > 0:
        st.dataframe(
            high_risk_customers[['Location Name', 'Core Industry', 'p_churn', 'business_risk_index', 'Churn or Retain']]
            .sort_values('business_risk_index', ascending=False)
            .head(20),
            use_container_width=True
        )

elif page == "Customer Segments":
    st.header("👥 Customer Segmentation Analysis")
    
    # Industry analysis
    st.subheader("🏭 Churn by Industry")
    
    industry_stats = scored_data.groupby('Core Industry').agg({
        'y_true': ['count', 'sum', 'mean'],
        'p_churn': 'mean',
        'business_risk_index': 'mean'
    }).round(3)
    
    industry_stats.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate', 'Avg_Churn_Prob', 'Avg_Risk_Index']
    industry_stats = industry_stats.reset_index()
    
    # Create industry visualization
    fig = px.bar(
        industry_stats,
        x='Core Industry',
        y='Churn_Rate',
        title="Churn Rate by Industry",
        labels={'Churn_Rate': 'Churn Rate', 'Core Industry': 'Industry'}
    )
    st.plotly_chart(fig, width='stretch')
    
    # Show industry table
    st.dataframe(industry_stats, width='stretch')
    
    # Bundle type analysis
    if 'BundleType' in scored_data.columns:
        st.subheader("📦 Churn by Bundle Type")
        
        bundle_stats = scored_data.groupby('BundleType').agg({
            'y_true': ['count', 'sum', 'mean'],
            'p_churn': 'mean'
        }).round(3)
        
        bundle_stats.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate', 'Avg_Churn_Prob']
        bundle_stats = bundle_stats.reset_index()
        
        fig = px.pie(
            bundle_stats,
            values='Total_Customers',
            names='BundleType',
            title="Customer Distribution by Bundle Type"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(bundle_stats, use_container_width=True)

elif page == "Churn Segmentation":
    st.header("🎯 Customer Churn Segmentation Analysis")
    
    # Load segmentation data
    @st.cache_data
    def load_segmentation_data():
        """Load customer segmentation results"""
        try:
            # Load segment info (summary data)
            segment_info_path = "0_6_month_churn_reason_segments_2024_segment_info.csv"
            if os.path.exists(segment_info_path):
                segment_info = pd.read_csv(segment_info_path)
            else:
                segment_info = None
            
            # Load segmented customer data
            segments_path = "0_6_month_churn_reason_segments_2024.csv"
            if os.path.exists(segments_path):
                segments_data = pd.read_csv(segments_path)
            else:
                segments_data = None
                
            return segment_info, segments_data
        except Exception as e:
            st.error(f"Error loading segmentation data: {str(e)}")
            return None, None
    
    segment_info, segments_data = load_segmentation_data()
    
    if segment_info is None and segments_data is None:
        st.warning("⚠️ Segmentation data not found. Please run the customer segmentation script first.")
        st.code("python customer_segmentation_0_6_month_churn.py", language="bash")
        st.stop()
    
    # Display segment overview
    if segment_info is not None:
        st.subheader("📊 Segment Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Segments", len(segment_info))
        with col2:
            if 'size' in segment_info.columns:
                total_customers = segment_info['size'].sum()
                st.metric("Total Customers", f"{total_customers:,}")
        with col3:
            if 'size' in segment_info.columns:
                avg_segment_size = segment_info['size'].mean()
                st.metric("Avg Segment Size", f"{avg_segment_size:.0f}")
        
        # Segment size distribution
        if 'size' in segment_info.columns:
            st.subheader("📈 Segment Size Distribution")
            fig = px.bar(
                segment_info,
                x='segment',
                y='size',
                title="Number of Customers by Segment",
                labels={'segment': 'Segment', 'size': 'Number of Customers'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment summaries
        st.subheader("📋 Segment Profiles")
        for _, row in segment_info.iterrows():
            with st.expander(f"Segment {row['segment']} ({row.get('size', 'N/A')} customers)"):
                if 'summary' in row and pd.notna(row['summary']):
                    st.markdown(row['summary'])
                else:
                    st.write("No summary available for this segment.")
    
    # Display customer data by segment
    if segments_data is not None:
        st.subheader("👥 Customer Details by Segment")
        
        # Segment filter
        available_segments = sorted(segments_data['reason_segment'].unique())
        selected_segment = st.selectbox(
            "Select Segment to View:",
            ["All Segments"] + [f"Segment {s}" for s in available_segments]
        )
        
        # Filter data based on selection
        if selected_segment != "All Segments":
            segment_num = int(selected_segment.split()[-1])
            filtered_segments = segments_data[segments_data['reason_segment'] == segment_num]
        else:
            filtered_segments = segments_data
        
        st.write(f"Showing {len(filtered_segments)} customers")
        
        # Key columns to display
        display_columns = [
            'LOCATION_NAME', 'CORE_INDUSTRY', 'PRACTICE_MANAGEMENT_SOFTWARE', 
            'STARTING_BUNDLE', 'LIFETIME_MONTHS', 'STARTING_MRR', 
            'SWAT_CANCEL_SUMMARY', 'reason_segment'
        ]
        
        # Only show columns that exist
        available_columns = [col for col in display_columns if col in filtered_segments.columns]
        
        # Display the data
        st.dataframe(
            filtered_segments[available_columns],
            use_container_width=True
        )
        
        # Download option
        csv = filtered_segments.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {selected_segment} Data as CSV",
            data=csv,
            file_name=f"churn_segmentation_{selected_segment.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        # Industry analysis by segment
        if 'CORE_INDUSTRY' in filtered_segments.columns:
            st.subheader("🏭 Industry Distribution in Selected Segment")
            industry_counts = filtered_segments['CORE_INDUSTRY'].value_counts()
            
            fig = px.pie(
                values=industry_counts.values,
                names=industry_counts.index,
                title=f"Industry Distribution - {selected_segment}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Display cluster visualization if available
    cluster_plot_path = "0_6_month_churn_cluster_plot.png"
    if os.path.exists(cluster_plot_path):
        st.subheader("🗺️ Cluster Visualization")
        st.image(cluster_plot_path, caption="2D PCA visualization of customer segments")

elif page == "AI Assistant":
    st.header("🤖 AI Assistant - Ask About Your Data")
    
    # Initialize OpenAI client for AI Assistant
    @st.cache_resource
    def get_openai_client():
        """Initialize OpenAI client for the AI assistant"""
        try:
            # Try Streamlit secrets first (for local development)
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            # Fallback to environment variable (for deployment)
            import os
            api_key = os.getenv("OPENAI_API_KEY")

        if api_key and api_key != "your-openai-api-key-here":
            return OpenAI(api_key=api_key), True
        else:
            return None, False
    
    ai_client, api_key_available = get_openai_client()
    
    if not api_key_available:
        st.warning("⚠️ OpenAI API key not found. Please add your API key to `.streamlit/secrets.toml` or set OPENAI_API_KEY environment variable")
        st.stop()
    
    # Set a default model for AI assistant
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    
    # Initialize chat history for AI assistant
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []
    
    # Add context about the data
    st.info("💡 **Tip**: Ask me about your churn prediction model, customer segments, risk analysis, or any insights from your data!")
    
    # Sample questions
    with st.expander("📝 Sample Questions You Can Ask"):
        st.markdown("""
        - "What are the key factors that predict customer churn?"
        - "Which customer segment has the highest risk?"
        - "How accurate is my churn prediction model?"
        - "What are the main reasons customers cancel?"
        - "Which industries have the highest churn rates?"
        - "How can I improve customer retention?"
        """)
    
    # Display chat messages from history on app rerun
    for message in st.session_state.ai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input for AI assistant
    if prompt := st.chat_input("Ask me anything about your churn data..."):
        if not api_key_available:
            st.error("Please configure your OpenAI API key to use the AI assistant.")
            st.stop()
            
        # Add user message to chat history
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response with context about the data
        with st.chat_message("assistant"):
            try:
                # Create context about the available data
                data_context = """
                You are an AI assistant helping analyze customer churn data for a company called Weave. 
                You have access to information about:
                - Customer churn prediction model results with features like Core Industry, Bundle Type, SWAT Cases, etc.
                - Customer segmentation analysis of churned customers
                - Risk analysis and business risk indices
                - Industry-wise churn patterns
                - Feature importance from logistic regression model
                
                The data includes customers from industries like Dental, Medical, Veterinary, and Optometry.
                Bundle types include various Weave product packages.
                Key features affecting churn include Week 1 Cases, Core Industry, Bundle Type, and various usage metrics.
                
                Please provide helpful, data-driven insights based on this context.
                """
                
                # Prepare messages with context
                messages_with_context = [
                    {"role": "system", "content": data_context}
                ] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.ai_messages
                ]
                
                response = ai_client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages_with_context,
                    stream=True,
                )
                
                # Stream the response
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                full_response = "Sorry, I encountered an error. Please try again."
            
            # Add assistant response to chat history
            st.session_state.ai_messages.append({"role": "assistant", "content": full_response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.ai_messages = []
        st.rerun()

elif page == "Detailed Results":
    st.header("📋 Detailed Model Results")
    
    # Filter options
    st.subheader("🔍 Filter Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        industry_filter = st.selectbox(
            "Select Industry:",
            ["All"] + list(scored_data['Core Industry'].unique())
        )
    
    with col2:
        risk_filter = st.selectbox(
            "Risk Level:",
            ["All", "High Risk (Top 20%)", "Medium Risk", "Low Risk (Bottom 20%)"]
        )
    
    with col3:
        churn_filter = st.selectbox(
            "Customer Status:",
            ["All", "Churned", "Retained"]
        )
    
    # Apply filters
    filtered_data = scored_data.copy()
    
    if industry_filter != "All":
        filtered_data = filtered_data[filtered_data['Core Industry'] == industry_filter]
    
    if risk_filter == "High Risk (Top 20%)":
        threshold = scored_data['business_risk_index'].quantile(0.8)
        filtered_data = filtered_data[filtered_data['business_risk_index'] >= threshold]
    elif risk_filter == "Low Risk (Bottom 20%)":
        threshold = scored_data['business_risk_index'].quantile(0.2)
        filtered_data = filtered_data[filtered_data['business_risk_index'] <= threshold]
    
    if churn_filter == "Churned":
        filtered_data = filtered_data[filtered_data['y_true'] == 1]
    elif churn_filter == "Retained":
        filtered_data = filtered_data[filtered_data['y_true'] == 0]
    
    st.write(f"Showing {len(filtered_data)} customers")
    
    # Display filtered results
    display_columns = [
        'Location Name', 'Core Industry', 'BundleType', 'Starting Mrr',
        'p_churn', 'business_risk_index', 'Churn or Retain'
    ]
    
    # Only show columns that exist
    available_columns = [col for col in display_columns if col in filtered_data.columns]
    
    st.dataframe(
        filtered_data[available_columns].sort_values('business_risk_index', ascending=False),
        use_container_width=True
    )
    
    # Download option
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Results as CSV",
        data=csv,
        file_name=f"filtered_churn_results_{len(filtered_data)}_customers.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("**Model Dashboard** | Built with Streamlit | Data-driven customer churn prediction")