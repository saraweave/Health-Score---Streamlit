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

# Weave logo at the top
try:
    st.image("weave logo.png", use_container_width=True)
except:
    st.write("**WEAVE**")  # Fallback if logo not found

st.title("Health Score - 0~6 months Single Churn Insights")

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
    ["Churned Customer Insights", "Customer Segments", "Feature Importance", "Risk Analysis"]
)

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

# Function to create AI assistant sidebar
def create_ai_assistant_sidebar(data_context_extra=""):
    """Create AI assistant in sidebar with optional data context"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🤖 AI Assistant")
        
        ai_client, api_key_available = get_openai_client()
        
        if not api_key_available:
            st.warning("⚠️ OpenAI API key not configured")
            return
        
        # Set a default model
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4o"
        
        # Initialize chat history
        if f"ai_messages_{page}" not in st.session_state:
            st.session_state[f"ai_messages_{page}"] = []
        
        # Chat input
        if prompt := st.chat_input("Ask about this section..."):
            # Add user message
            st.session_state[f"ai_messages_{page}"].append({"role": "user", "content": prompt})
            
            # Create context based on current page
            context_map = {
                "Churned Customer Insights": "You are analyzing churned customer segments and reasons for cancellation.",
                "Customer Segments": "You are analyzing customer segmentation by industry, bundle type, and other characteristics.",
                "Feature Importance": "You are analyzing feature importance from the churn prediction model and which factors most influence churn.",
                "Risk Analysis": "You are analyzing customer risk levels and business risk indices for churn prediction."
            }
            
            data_context = f"""
            You are an AI assistant helping analyze customer churn data for Weave. 
            Current focus: {context_map.get(page, "General churn analysis")}
            
            The data includes customers from industries like Dental, Medical, Veterinary, and Optometry.
            Key features include Week 1 Cases, Core Industry, Bundle Type, and various usage metrics.
            
            {data_context_extra}
            
            Provide concise, actionable insights based on this data.
            """
            
            try:
                messages_with_context = [
                    {"role": "system", "content": data_context}
                ] + st.session_state[f"ai_messages_{page}"]
                
                response = ai_client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages_with_context,
                    stream=False,
                    max_tokens=300
                )
                
                assistant_response = response.choices[0].message.content
                st.session_state[f"ai_messages_{page}"].append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                st.error(f"AI Error: {str(e)}")
        
        # Display recent messages
        recent_messages = st.session_state[f"ai_messages_{page}"][-4:]  # Show last 2 exchanges
        for message in recent_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if st.button("Clear Chat", key=f"clear_{page}"):
            st.session_state[f"ai_messages_{page}"] = []
            st.rerun()

if page == "Churned Customer Insights":
    st.header("🎯 Churned Customer Insights & Segmentation")
    
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
    
    # Create comprehensive data context for AI assistant from segment info
    ai_data_context = ""
    if segment_info is not None:
        total_segments = len(segment_info)
        if 'size' in segment_info.columns:
            total_customers = segment_info['size'].sum()
            avg_segment_size = segment_info['size'].mean()
            ai_data_context += f"""
            CHURN SEGMENTATION ANALYSIS (0-6 Month Customers):
            - Total segments identified: {total_segments}
            - Total churned customers analyzed: {total_customers:,}
            - Average segment size: {avg_segment_size:.0f} customers
            
            DETAILED SEGMENT PROFILES:
            """
            
            # Add full segment summaries from the CSV
            if 'summary' in segment_info.columns:
                for _, row in segment_info.iterrows():
                    if pd.notna(row.get('summary')):
                        ai_data_context += f"""
            SEGMENT {row['segment']} ({row.get('size', 'N/A')} customers):
            {row['summary']}
            
            """
    
    if segments_data is not None:
        # Add comprehensive data analysis
        total_customers = len(segments_data)
        ai_data_context += f"""
        DETAILED CUSTOMER-LEVEL DATA ({total_customers:,} customers):
        """
        
        # Industry breakdown
        if 'CORE_INDUSTRY' in segments_data.columns:
            industry_counts = segments_data['CORE_INDUSTRY'].value_counts()
            ai_data_context += f"\nINDUSTRY BREAKDOWN:\n"
            for industry, count in industry_counts.head(5).items():
                pct = (count / total_customers * 100)
                ai_data_context += f"- {industry}: {count} customers ({pct:.1f}%)\n"
        
        # Bundle analysis
        if 'STARTING_BUNDLE' in segments_data.columns:
            bundle_counts = segments_data['STARTING_BUNDLE'].value_counts()
            ai_data_context += f"\nTOP STARTING BUNDLES:\n"
            for bundle, count in bundle_counts.head(5).items():
                if pd.notna(bundle):
                    ai_data_context += f"- {bundle}: {count} customers\n"
        
        # Lifetime analysis
        if 'LIFETIME_MONTHS' in segments_data.columns:
            avg_lifetime = segments_data['LIFETIME_MONTHS'].mean()
            median_lifetime = segments_data['LIFETIME_MONTHS'].median()
            ai_data_context += f"\nTENURE ANALYSIS:\n- Average lifetime: {avg_lifetime:.1f} months\n- Median lifetime: {median_lifetime:.1f} months\n"
        
        # MRR analysis
        if 'STARTING_MRR' in segments_data.columns:
            avg_mrr = segments_data['STARTING_MRR'].mean()
            total_lost_mrr = segments_data['STARTING_MRR'].sum()
            ai_data_context += f"\nREVENUE IMPACT:\n- Average MRR per customer: ${avg_mrr:.2f}\n- Total lost MRR: ${total_lost_mrr:,.2f}\n"
        
        # PMS analysis
        if 'PRACTICE_MANAGEMENT_SOFTWARE' in segments_data.columns:
            pms_counts = segments_data['PRACTICE_MANAGEMENT_SOFTWARE'].value_counts()
            ai_data_context += f"\nTOP PRACTICE MANAGEMENT SYSTEMS:\n"
            for pms, count in pms_counts.head(5).items():
                if pd.notna(pms) and pms != '':
                    ai_data_context += f"- {pms}: {count} customers\n"
        
        # Detailed customer examples by segment for AI reference
        ai_data_context += f"\nCUSTOMER EXAMPLES BY SEGMENT:\n"
        for segment in segments_data['reason_segment'].unique():
            segment_customers = segments_data[segments_data['reason_segment'] == segment]
            ai_data_context += f"\nSEGMENT {segment} EXAMPLES:\n"
            
            # Add specific customer examples with key details
            sample_customers = segment_customers.head(3)
            for idx, (_, customer) in enumerate(sample_customers.iterrows(), 1):
                location_name = customer.get('LOCATION_NAME', 'N/A')
                industry = customer.get('CORE_INDUSTRY', 'N/A')
                bundle = customer.get('STARTING_BUNDLE', 'N/A')
                mrr = customer.get('STARTING_MRR', 0)
                lifetime = customer.get('LIFETIME_MONTHS', 0)
                pms = customer.get('PRACTICE_MANAGEMENT_SOFTWARE', 'N/A')
                cancel_reason = customer.get('SWAT_CANCEL_SUMMARY', 'N/A')
                
                ai_data_context += f"  {idx}. {location_name} ({industry}) - ${mrr:.0f} MRR, {lifetime} months, {pms} PMS\n"
                if pd.notna(cancel_reason) and cancel_reason != 'N/A':
                    ai_data_context += f"     Reason: {cancel_reason[:200]}...\n"
        
        # Specific customer needs and pain points by industry
        ai_data_context += f"\nCUSTOMER NEEDS BY INDUSTRY:\n"
        for industry in segments_data['CORE_INDUSTRY'].value_counts().head(4).index:
            industry_customers = segments_data[segments_data['CORE_INDUSTRY'] == industry]
            ai_data_context += f"\n{industry.upper()} CUSTOMERS ({len(industry_customers)} total):\n"
            
            # Common PMS systems
            top_pms = industry_customers['PRACTICE_MANAGEMENT_SOFTWARE'].value_counts().head(3)
            ai_data_context += f"  Common PMS: {', '.join([f'{pms} ({count})' for pms, count in top_pms.items() if pd.notna(pms)])}\n"
            
            # Common bundles
            top_bundles = industry_customers['STARTING_BUNDLE'].value_counts().head(2)
            ai_data_context += f"  Popular Bundles: {', '.join([f'{bundle}' for bundle in top_bundles.index if pd.notna(bundle)])}\n"
            
            # Average metrics
            avg_mrr = industry_customers['STARTING_MRR'].mean()
            avg_lifetime = industry_customers['LIFETIME_MONTHS'].mean()
            ai_data_context += f"  Avg MRR: ${avg_mrr:.0f}, Avg Lifetime: {avg_lifetime:.1f} months\n"
        
        # Common pain points and solutions needed
        ai_data_context += f"\nCOMMON CUSTOMER PAIN POINTS & NEEDS:\n"
        if 'SWAT_CANCEL_SUMMARY' in segments_data.columns:
            cancel_reasons = segments_data[segments_data['SWAT_CANCEL_SUMMARY'].notna()]['SWAT_CANCEL_SUMMARY']
            
            # Extract common themes
            integration_issues = len([r for r in cancel_reasons if pd.notna(r) and any(word in r.lower() for word in ['integration', 'sync', 'pms', 'data'])])
            support_issues = len([r for r in cancel_reasons if pd.notna(r) and any(word in r.lower() for word in ['support', 'onboarding', 'training', 'help'])])
            technical_issues = len([r for r in cancel_reasons if pd.notna(r) and any(word in r.lower() for word in ['technical', 'bug', 'system', 'error', 'problem'])])
            pricing_issues = len([r for r in cancel_reasons if pd.notna(r) and any(word in r.lower() for word in ['price', 'cost', 'expensive', 'pricing'])])
            
            ai_data_context += f"- Integration/Data Sync Issues: {integration_issues} customers\n"
            ai_data_context += f"- Support/Onboarding Issues: {support_issues} customers\n"
            ai_data_context += f"- Technical Issues: {technical_issues} customers\n"
            ai_data_context += f"- Pricing Concerns: {pricing_issues} customers\n"
        
        # Cluster visualization methodology explanation
        ai_data_context += f"\nCLUSTER VISUALIZATION METHODOLOGY:\n"
        ai_data_context += "The 2D PCA visualization shows customer segments created using advanced machine learning:\n\n"
        ai_data_context += "EMBEDDING PROCESS:\n"
        ai_data_context += "- Job Characteristics: 12-dimensional embeddings from Core Industry, PMS, Integrations, Bundle, Lifetime\n"
        ai_data_context += "- Cancellation Reasons: 24-dimensional embeddings from SWAT_CANCEL_SUMMARY text\n"
        ai_data_context += "- Total: 36-dimensional feature space combining customer profile + cancellation feedback\n"
        ai_data_context += "- Model: OpenAI text-embedding-3-small for semantic understanding\n\n"
        ai_data_context += "CLUSTERING ALGORITHM:\n"
        ai_data_context += "- Method: KMeans clustering with 3 segments (optimized for interpretability)\n"
        ai_data_context += "- Random State: 42 (for reproducible results)\n"
        ai_data_context += "- Input: 36-dimensional embedded customer vectors\n\n"
        ai_data_context += "VISUALIZATION TECHNIQUE:\n"
        ai_data_context += "- Dimensionality Reduction: PCA (Principal Component Analysis) to 2D\n"
        ai_data_context += "- X-axis: Principal Component 1 (captures most variance in customer differences)\n"
        ai_data_context += "- Y-axis: Principal Component 2 (captures second most important differences)\n"
        ai_data_context += "- Colors: Each segment shown in different color for easy identification\n\n"
        ai_data_context += "INTERPRETATION GUIDE:\n"
        ai_data_context += "- Proximity: Customers close together have similar profiles and cancellation reasons\n"
        ai_data_context += "- Separation: Distance between clusters shows how distinct the segments are\n"
        ai_data_context += "- Scatter: Tight clusters = homogeneous segments, spread out = diverse segments\n"
        ai_data_context += "- Each point represents one churned customer positioned by their combined profile + reason\n\n"
        
        # Store full dataset for AI to reference specific customers
        ai_data_context += f"COMPREHENSIVE DATA ACCESS:\n"
        ai_data_context += f"I have access to detailed information for all {len(segments_data)} customers including:\n"
        ai_data_context += "- Location names, industries, and specialties\n"
        ai_data_context += "- MRR values, lifetime, and bundle types\n"
        ai_data_context += "- PMS systems and integration details\n"
        ai_data_context += "- Specific cancellation reasons and feedback\n"
        ai_data_context += "- Segment classifications and usage patterns\n"
        ai_data_context += "- Clustering methodology and visualization interpretation\n"
        ai_data_context += "I can provide specific customer lists, explain clustering decisions, and interpret the PCA visualization.\n"
    
    # Store customer data for AI assistant access
    if segments_data is not None:
        st.session_state['customer_segments_data'] = segments_data
    
    # Add AI assistant to sidebar with data context
    create_ai_assistant_sidebar(ai_data_context)
    
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
        
        # Display cluster visualization if available
        cluster_plot_path = "0_6_month_churn_cluster_plot.png"
        if os.path.exists(cluster_plot_path):
            st.subheader("🗺️ Cluster Visualization")
            st.image(cluster_plot_path, caption="2D PCA visualization of customer segments")
        
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

elif page == "Risk Analysis":
    # Add AI assistant to sidebar
    create_ai_assistant_sidebar()
    
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
    # Add AI assistant to sidebar
    create_ai_assistant_sidebar()
    
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

elif page == "Feature Importance":
    # Add AI assistant to sidebar
    create_ai_assistant_sidebar()
    
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
        use_container_width=True
    )

# Footer
st.markdown("---")