import streamlit as st
import fitz  # PyMuPDF
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from fpdf import FPDF
import networkx as nx
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings, CohereEmbeddings
from langchain_community.llms import LlamaCpp
import numpy as np
import openai
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from graphviz import Digraph

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use FakeEmbeddings for demo/testing
embeddings = FakeEmbeddings(size=1536)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

print(nltk.data.path)

def generate_concept_map(docs, metas):
    """Generate a concept map showing relationships between papers with improved similarity calculations."""
    try:
        # Group all chunks by paper
        paper_texts = {}
        paper_metadata = {}
        for chunk, meta in zip(docs, metas):
            paper = meta["source"]
            if paper not in paper_texts:
                paper_texts[paper] = []
                paper_metadata[paper] = {"chunks": [], "sections": {}}
            paper_texts[paper].append(chunk)
            paper_metadata[paper]["chunks"].append(chunk)
        
        # Process each paper's text
        papers = list(paper_texts.keys())
        st.write(f"Found {len(papers)} unique papers")
        
        # Create a graph
        G = nx.Graph()
        
        # Add paper nodes with metadata
        for paper in papers:
            # Extract basic metadata from filename
            year = re.search(r'\b(20|19)\d{2}\b', paper)
            year = year.group(0) if year else "Unknown"
            G.add_node(paper, 
                      type="paper",
                      year=year,
                      size=len(paper_texts[paper]))  # Node size based on content length
        
        # Calculate similarities between papers
        edge_count = 0
        for i in range(len(papers)):
            for j in range(i+1, len(papers)):
                paper1, paper2 = papers[i], papers[j]
                text1 = " ".join(paper_texts[paper1])
                text2 = " ".join(paper_texts[paper2])
                
                # Calculate multiple similarity metrics
                similarity = calculate_document_similarity(text1, text2)
                
                # Add edge if similarity is above threshold
                if similarity > 0.1:  # Adjust threshold as needed
                    G.add_edge(paper1, paper2, 
                             weight=similarity,
                             type="similarity")
                    edge_count += 1
        
        st.write(f"Generated graph with {len(G.nodes())} nodes and {edge_count} edges")
        
        # Add node attributes for visualization
        for node in G.nodes():
            # Calculate node importance based on connections
            importance = len(list(G.neighbors(node)))
            G.nodes[node]['importance'] = importance
            
            # Add year information for coloring
            year = G.nodes[node].get('year', 'Unknown')
            G.nodes[node]['year'] = year
        
        return G
    except Exception as e:
        st.error(f"Error generating concept map: {str(e)}")
        return nx.Graph()  # Return empty graph on error

def calculate_document_similarity(doc1, doc2):
    """Calculate similarity between two documents using embeddings."""
    try:
        # Get embeddings for both documents
        emb1 = embeddings.embed_query(doc1)
        emb2 = embeddings.embed_query(doc2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    except Exception as e:
        st.warning(f"Error calculating similarity: {str(e)}")
        return 0.0

def visualize_concept_map(G):
    """Convert networkx graph to plotly format with improved visualization."""
    try:
        # Create a layout for the graph
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_weights = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 0.5))

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create hover text with metadata
            year = G.nodes[node].get('year', 'Unknown')
            importance = G.nodes[node].get('importance', 0)
            hover_text = f"{node}<br>Year: {year}<br>Connections: {importance}"
            node_text.append(hover_text)
            
            # Size based on importance
            node_sizes.append(20 + importance * 5)
            
            # Color based on year
            node_colors.append(importance)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_sizes,
                color=node_colors,
                colorbar=dict(
                    thickness=15,
                    title='Connections',
                    xanchor='left'
                )
            ))

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title=dict(
                             text='Paper Relationship Map',
                             font=dict(size=16)
                         ),
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                     )

        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Add legend
        st.markdown("""
        **Map Legend:**
        - 🔵 Nodes: Individual papers
        - ⚪ Edges: Relationships between papers
        - Node size: Number of connections
        - Node color: Connection strength
        """)
        
    except Exception as e:
        st.error(f"Error displaying concept map: {str(e)}")
        st.info("Please try refreshing the page or rebuilding the knowledge base.")

SECTIONS = ["abstract", "methods", "results", "discussion", "limitations"]

# === PDF Utilities ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def extract_sections_and_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])

    # Extract metadata
    title = doc.metadata.get("title", "Unknown Title")
    authors = doc.metadata.get("author", "Unknown Author")
    year = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    year = year.group(0) if year else "Unknown Year"

    # Clean the text
    # Remove URLs and DOIs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'doi:[\d\.]+/[\w\.-]+', '', text)
    # Remove common metadata headers
    text = re.sub(r'Peer review under responsibility of.*?\.', '', text)
    text = re.sub(r'Production and hosting by.*?\.', '', text)
    text = re.sub(r'Contents.*?\.', '', text)
    text = re.sub(r'www\.[\w\.-]+\.com', '', text)
    text = re.sub(r'www\.[\w\.-]+\.edu', '', text)
    text = re.sub(r'©.*?\.', '', text)
    text = re.sub(r'All rights reserved.*?\.', '', text)
    text = re.sub(r'\d{4}.*?University.*?\.', '', text)

    # Define section patterns with more variations and context
    section_patterns = {
        'abstract': r'(?i)(?:^|\n)\s*(?:abstract|summary|overview|synopsis)(?:\s*:|\s*$|\s*\n)',
        'introduction': r'(?i)(?:^|\n)\s*(?:introduction|background|overview|context|purpose|aims?|objectives?)(?:\s*:|\s*$|\s*\n)',
        'methods': r'(?i)(?:^|\n)\s*(?:methods?|methodology|experimental|materials and methods?|synthesis|preparation|fabrication|procedures?|techniques?|approach|experimental section|materials|experimental details?|experimental procedure|experimental setup|experimental design|experimental protocol|experimental conditions?|experimental parameters?)(?:\s*:|\s*$|\s*\n)',
        'results': r'(?i)(?:^|\n)\s*(?:results?|findings?|experimental results?|characterization|analysis|measurements?|observations?|data|experimental data|experimental findings?|experimental observations?|experimental analysis|experimental measurements?)(?:\s*:|\s*$|\s*\n)',
        'discussion': r'(?i)(?:^|\n)\s*(?:discussion|analysis|interpretation|implications|applications?|significance|implications?|conclusions?|findings?|results?|analysis and discussion|discussion and conclusions?)(?:\s*:|\s*$|\s*\n)',
        'conclusion': r'(?i)(?:^|\n)\s*(?:conclusion|conclusions?|summary|final remarks|closing|concluding remarks|final thoughts|summary and conclusions?)(?:\s*:|\s*$|\s*\n)',
        'limitations': r'(?i)(?:^|\n)\s*(?:limitations?|constraints?|challenges?|future work|perspectives?|outlook|future directions?|future perspectives?|future challenges?|future opportunities?|future developments?)(?:\s*:|\s*$|\s*\n)'
    }

    # First pass: Find all potential section headers
    section_matches = []
    for section_name, pattern in section_patterns.items():
        matches = list(re.finditer(pattern, text))
        for match in matches:
            # Get some context around the match
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            # Only add if it looks like a real section header
            if not any(term in context.lower() for term in ['table', 'figure', 'fig.', 'ref.', 'reference']):
                section_matches.append((match.start(), section_name, match))

    # Sort matches by position
    section_matches.sort(key=lambda x: x[0])

    # Extract sections
    sections = {}
    for i, (start, section_name, match) in enumerate(section_matches):
        # Get the end position (start of next section or end of text)
        end = section_matches[i + 1][0] if i + 1 < len(section_matches) else len(text)
        
        # Extract and clean section text
        section_text = text[match.end():end].strip()
        
        # Clean section text
        section_text = re.sub(r'\n+', ' ', section_text)  # Replace multiple newlines with space
        section_text = re.sub(r'\s+', ' ', section_text)  # Replace multiple spaces with single space
        section_text = re.sub(r'^\d+\.\s*', '', section_text)  # Remove numbered lists at start
        
        # Remove figure and table references
        section_text = re.sub(r'Figure \d+.*?\.', '', section_text)
        section_text = re.sub(r'Table \d+.*?\.', '', section_text)
        section_text = re.sub(r'Fig\. \d+.*?\.', '', section_text)
        
        # Clean up citations while preserving the text
        section_text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', section_text)  # Remove citation numbers
        section_text = re.sub(r'\(\d{4}\)', '', section_text)  # Remove year citations
        
        # Only add non-empty sections
        if section_text and len(section_text.split()) > 10:  # Ensure section has meaningful content
            sections[section_name] = section_text

    # If no sections were found or sections are too short, try to split the text into logical sections
    if not sections or all(len(text.split()) < 50 for text in sections.values()):
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        current_section = 'introduction'
        sections[current_section] = ''
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Check if paragraph starts with a section header
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, para):
                    current_section = section_name
                    para = re.sub(pattern, '', para).strip()
                    break
            
            # Skip paragraphs that are just figure/table references
            if re.match(r'^(Figure|Table|Fig\.)\s+\d+', para):
                continue
                
            if para and len(para.split()) > 5:  # Only add paragraphs with meaningful content
                if current_section in sections:
                    sections[current_section] += ' ' + para
                else:
                    sections[current_section] = para

    # Clean up sections
    for section_name in list(sections.keys()):
        text = sections[section_name]
        # Remove any remaining figure/table references
        text = re.sub(r'Figure \d+.*?\.', '', text)
        text = re.sub(r'Table \d+.*?\.', '', text)
        text = re.sub(r'Fig\. \d+.*?\.', '', text)
        # Remove any remaining metadata
        text = re.sub(r'©.*?\.', '', text)
        text = re.sub(r'All rights reserved.*?\.', '', text)
        # Clean up citations
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        text = re.sub(r'\(\d{4}\)', '', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text.split()) > 10:  # Only keep sections with meaningful content
            sections[section_name] = text
        else:
            del sections[section_name]

    # Ensure all required sections exist and have meaningful content
    required_sections = ['abstract', 'methods', 'results', 'discussion', 'limitations']
    for section in required_sections:
        if section not in sections or len(sections[section].split()) < 10:
            # Try to find content for missing sections
            if section == 'abstract':
                # Look for content at the beginning of the text, before any section headers
                first_section_start = min([match.start() for _, _, match in section_matches]) if section_matches else len(text)
                abstract_text = text[:first_section_start].strip()
                
                # Split into paragraphs and take the first few meaningful ones
                paragraphs = [p.strip() for p in abstract_text.split('\n\n') if p.strip()]
                abstract_content = []
                for para in paragraphs[:3]:  # Take first 3 paragraphs
                    if len(para.split()) > 5 and not any(keyword in para.lower() for keyword in 
                        ['method', 'procedure', 'technique', 'experiment', 'materials', 'synthesis']):
                        abstract_content.append(para)
                
                if abstract_content:
                    sections[section] = ' '.join(abstract_content)
                else:
                    sections[section] = "No specific abstract content found in the text."
            elif section == 'methods':
                # Look for content about experimental procedures, materials, and techniques
                methods_keywords = [
                    'method', 'procedure', 'technique', 'protocol', 'experiment', 'experimental',
                    'synthesis', 'preparation', 'fabrication', 'manufacturing', 'processing',
                    'material', 'sample', 'specimen', 'substrate', 'solution', 'mixture',
                    'reagent', 'chemical', 'compound', 'polymer', 'nanoparticle', 'nanotube',
                    'characterization', 'analysis', 'measurement', 'testing', 'evaluation',
                    'instrument', 'equipment', 'apparatus', 'setup', 'configuration',
                    'condition', 'parameter', 'temperature', 'pressure', 'concentration',
                    'time', 'duration', 'rate', 'speed', 'flow', 'volume', 'weight',
                    'prepared', 'synthesized', 'fabricated', 'manufactured', 'processed',
                    'analyzed', 'measured', 'characterized', 'tested', 'evaluated',
                    'using', 'with', 'by', 'through', 'via', 'employing', 'utilizing'
                ]
                methods_content = []
                for sentence in text.split('.'):
                    # Check if sentence contains methods-related keywords and is not from abstract
                    if any(keyword in sentence.lower() for keyword in methods_keywords):
                        # Skip if sentence is likely from abstract
                        if not any(keyword in sentence.lower() for keyword in 
                            ['study', 'investigate', 'examine', 'analyze', 'report', 'present']):
                            methods_content.append(sentence.strip())
                if methods_content:
                    sections[section] = '. '.join(methods_content) + '.'
                else:
                    sections[section] = "No specific methods content found in the text."
            elif section == 'results':
                # Look for content about findings, measurements, or observations
                results_keywords = [
                    'found', 'observed', 'measured', 'detected', 'showed', 'demonstrated',
                    'revealed', 'indicated', 'determined', 'calculated', 'estimated',
                    'analyzed', 'characterized', 'studied', 'investigated', 'examined',
                    'evaluated', 'assessed', 'tested', 'verified', 'confirmed', 'validated',
                    'result', 'finding', 'measurement', 'observation', 'data', 'analysis',
                    'value', 'parameter', 'property', 'characteristic', 'feature',
                    'pattern', 'trend', 'correlation', 'relationship', 'dependence',
                    'effect', 'influence', 'impact', 'change', 'variation', 'difference',
                    'increase', 'decrease', 'improvement', 'enhancement', 'reduction',
                    'significant', 'substantial', 'considerable', 'notable', 'remarkable',
                    'consistent', 'reproducible', 'reliable', 'accurate', 'precise'
                ]
                results_content = []
                for sentence in text.split('.'):
                    if any(keyword in sentence.lower() for keyword in results_keywords):
                        results_content.append(sentence.strip())
                if results_content:
                    sections[section] = '. '.join(results_content) + '.'
                else:
                    sections[section] = "No specific results content found in the text."
            elif section == 'discussion':
                # Look for content about implications, significance, or applications
                discussion_keywords = [
                    'implications', 'significance', 'applications', 'importance',
                    'relevance', 'impact', 'influence', 'effect', 'role', 'contribution',
                    'advancement', 'development', 'progress', 'improvement', 'enhancement',
                    'optimization', 'refinement', 'suggest', 'indicate', 'demonstrate',
                    'show', 'prove', 'confirm', 'verify', 'validate', 'support',
                    'consistent', 'agreement', 'accordance', 'compatibility', 'alignment',
                    'comparison', 'contrast', 'difference', 'similarity', 'analogy',
                    'advantage', 'benefit', 'merit', 'strength', 'potential',
                    'limitation', 'constraint', 'challenge', 'issue', 'concern',
                    'future', 'prospect', 'opportunity', 'direction', 'perspective'
                ]
                discussion_content = []
                for sentence in text.split('.'):
                    if any(keyword in sentence.lower() for keyword in discussion_keywords):
                        discussion_content.append(sentence.strip())
                if discussion_content:
                    sections[section] = '. '.join(discussion_content) + '.'
                else:
                    sections[section] = "No specific discussion content found in the text."
            elif section == 'limitations':
                # Look for content about limitations, challenges, or future work
                limitations_keywords = [
                    'limitations', 'constraints', 'challenges', 'difficulties',
                    'problems', 'issues', 'concerns', 'drawbacks', 'shortcomings',
                    'weaknesses', 'deficiencies', 'gaps', 'needs', 'requirements',
                    'future', 'prospects', 'potential', 'opportunities', 'improve',
                    'enhance', 'develop', 'advance', 'optimize', 'refine',
                    'overcome', 'address', 'resolve', 'mitigate', 'minimize',
                    'reduce', 'eliminate', 'prevent', 'avoid', 'circumvent',
                    'alternative', 'substitute', 'replacement', 'modification',
                    'adjustment', 'adaptation', 'improvement', 'enhancement',
                    'development', 'advancement', 'progress', 'evolution'
                ]
                limitations_content = []
                for sentence in text.split('.'):
                    if any(keyword in sentence.lower() for keyword in limitations_keywords):
                        limitations_content.append(sentence.strip())
                if limitations_content:
                    sections[section] = '. '.join(limitations_content) + '.'
                else:
                    sections[section] = "No specific limitations content found in the text."
            else:
                sections[section] = "No content available for this section."

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "sections": sections
    }

def simple_summarize(text, max_length=1500):
    """Enhanced local summarization with improved text cleaning and sentence selection."""
    try:
        # Clean the text
        # Remove URLs and DOIs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'doi:[\d\.]+/[\w\.-]+', '', text)
        # Remove common metadata headers
        text = re.sub(r'Peer review under responsibility of.*?\.', '', text)
        text = re.sub(r'Production and hosting by.*?\.', '', text)
        text = re.sub(r'Contents.*?\.', '', text)
        text = re.sub(r'www\.[\w\.-]+\.com', '', text)
        text = re.sub(r'www\.[\w\.-]+\.edu', '', text)
        text = re.sub(r'©.*?\.', '', text)
        text = re.sub(r'All rights reserved.*?\.', '', text)
        text = re.sub(r'\d{4}.*?University.*?\.', '', text)
        
        # Split text into sentences using NLTK
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to basic sentence splitting if NLTK fails
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Filter and score sentences
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence.split()) < 4:
                continue
                
            # Skip sentences that are likely metadata or headers
            if any(term in sentence.lower() for term in ['www.', 'http', 'doi:', 'copyright', 'all rights reserved']):
                continue
                
            # Score sentence based on various factors
            score = 0
            
            # Prioritize first and last sentences
            if i == 0 or i == 1:
                score += 4  # prioritize intro
            if i == len(sentences) - 1 or i == len(sentences) - 2:
                score += 4  # prioritize conclusion
                
            # Prefer sentences that start with key phrases
            if any(sentence.lower().startswith(phrase) for phrase in 
                  ['the study', 'we found', 'results show', 'this research', 'our analysis', 
                   'the authors', 'the paper', 'the findings', 'this paper', 'we demonstrate',
                   'the results', 'the analysis', 'the experiment', 'the method', 'in this',
                   'this work', 'we present', 'we report', 'we investigate', 'we examine',
                   'the research', 'our study', 'this study', 'we studied', 'we analyzed',
                   'the investigation', 'our findings', 'the experiment', 'the analysis',
                   'the research shows', 'the study demonstrates', 'we observed',
                   'the method', 'the procedure', 'the technique', 'the approach',
                   'the experiment', 'the analysis', 'the measurement', 'the observation',
                   'the finding', 'the result', 'the discovery', 'the observation',
                   'the conclusion', 'the implication', 'the significance', 'the impact',
                   'the limitation', 'the challenge', 'the future', 'the prospect']):
                score += 4
                
            # Prefer sentences with key terms
            if any(term in sentence.lower() for term in 
                  ['significant', 'important', 'key', 'main', 'primary', 'conclude', 
                   'demonstrate', 'indicate', 'suggest', 'reveal', 'show', 'prove',
                   'evidence', 'finding', 'result', 'analysis', 'method', 'approach',
                   'technique', 'procedure', 'experiment', 'study', 'research',
                   'observe', 'determine', 'establish', 'identify', 'characterize',
                   'develop', 'investigate', 'examine', 'analyze', 'measure',
                   'observe', 'demonstrate', 'confirm', 'verify', 'validate',
                   'compare', 'contrast', 'evaluate', 'assess', 'examine',
                   'investigate', 'explore', 'study', 'analyze', 'measure',
                   'method', 'procedure', 'technique', 'protocol', 'experiment',
                   'synthesis', 'preparation', 'fabrication', 'manufacturing',
                   'material', 'sample', 'specimen', 'substrate', 'solution',
                   'reagent', 'chemical', 'compound', 'polymer', 'nanoparticle',
                   'characterization', 'analysis', 'measurement', 'testing',
                   'instrument', 'equipment', 'apparatus', 'setup', 'configuration',
                   'condition', 'parameter', 'temperature', 'pressure', 'concentration',
                   'time', 'duration', 'rate', 'speed', 'flow', 'volume', 'weight',
                   'prepared', 'synthesized', 'fabricated', 'manufactured', 'processed',
                   'analyzed', 'measured', 'characterized', 'tested', 'evaluated',
                   'using', 'with', 'by', 'through', 'via', 'employing', 'utilizing',
                   'found', 'observed', 'measured', 'detected', 'showed', 'demonstrated',
                   'revealed', 'indicated', 'determined', 'calculated', 'estimated',
                   'analyzed', 'characterized', 'studied', 'investigated', 'examined',
                   'evaluated', 'assessed', 'tested', 'verified', 'confirmed', 'validated',
                   'result', 'finding', 'measurement', 'observation', 'data', 'analysis',
                   'value', 'parameter', 'property', 'characteristic', 'feature',
                   'pattern', 'trend', 'correlation', 'relationship', 'dependence',
                   'effect', 'influence', 'impact', 'change', 'variation', 'difference',
                   'increase', 'decrease', 'improvement', 'enhancement', 'reduction',
                   'significant', 'substantial', 'considerable', 'notable', 'remarkable',
                   'consistent', 'reproducible', 'reliable', 'accurate', 'precise',
                   'implications', 'significance', 'applications', 'importance',
                   'relevance', 'impact', 'influence', 'effect', 'role', 'contribution',
                   'advancement', 'development', 'progress', 'improvement', 'enhancement',
                   'optimization', 'refinement', 'suggest', 'indicate', 'demonstrate',
                   'show', 'prove', 'confirm', 'verify', 'validate', 'support',
                   'consistent', 'agreement', 'accordance', 'compatibility', 'alignment',
                   'comparison', 'contrast', 'difference', 'similarity', 'analogy',
                   'advantage', 'benefit', 'merit', 'strength', 'potential',
                   'limitation', 'constraint', 'challenge', 'issue', 'concern',
                   'future', 'prospect', 'opportunity', 'direction', 'perspective']):
                score += 3
                
            # Prefer sentences with numbers (often contain important data)
            if any(c.isdigit() for c in sentence):
                score += 3
                
            # Penalize very long sentences
            if len(sentence.split()) > 30:
                score -= 1
                
            scored_sentences.append((sentence, score))
        
        # Sort sentences by score and select top ones
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected_sentences = [s[0] for s in scored_sentences[:30]]  # Take top 30 sentences
        
        # Organize sentences into a coherent summary
        summary_parts = []
        
        # Add introduction (first few high-scoring sentences)
        intro_sentences = [s for s in selected_sentences if s in sentences[:5]]
        if intro_sentences:
            summary_parts.append(" ".join(intro_sentences))
        
        # Add main content (remaining selected sentences)
        main_sentences = [s for s in selected_sentences if s not in intro_sentences]
        if main_sentences:
            summary_parts.append(" ".join(main_sentences))
        
        # Combine parts
        summary = " ".join(summary_parts)
        
        # Ensure summary doesn't exceed max_length
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
            
        return summary
            
    except Exception as e:
        st.info(f"Using basic summarization method due to error: {str(e)}")
        # Last resort fallback
        return text[:max_length] + "..."

def summarize_section(section_text, section_name, openai_api_key):
    """Summarize a section using OpenAI API with improved fallback to local summarization."""
    if not openai_api_key:
        return simple_summarize(section_text)
        
    try:
        prompt = f"Summarize the following {section_name} section in 2 sentences:\n\n{section_text[:2000]}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        time.sleep(1)  # Add a 1-second delay between calls
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Use enhanced local summarization
        return simple_summarize(section_text)

# === Document Summarization ===
def run_deep_analysis(text):
    """Run deep analysis using simple prompts."""
    try:
        # Split text into chunks for processing
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        
        # Define analysis prompts
        prompts = {
            "Abstract": "Analyze the abstract and provide:\n1. Main research objectives\n2. Key methodology\n3. Major findings\n4. Significance of the work",
            
            "Methods": "Analyze the methods section and provide:\n1. Experimental approach\n2. Materials and techniques used\n3. Key procedures\n4. Analytical methods",
            
            "Results": "Analyze the results section and provide:\n1. Main findings\n2. Key data points\n3. Important observations\n4. Technical outcomes",
            
            "Discussion": "Analyze the discussion section and provide:\n1. Interpretation of results\n2. Implications\n3. Comparison with existing work\n4. Future implications",
            
            "Limitations": "Analyze the limitations and provide:\n1. Current limitations\n2. Technical challenges\n3. Future improvements needed\n4. Research gaps"
        }
        
        summaries = {}
        
        # Process each section
        for section, prompt in prompts.items():
            with st.spinner(f"Analyzing {section}..."):
                # Use local summarization with the prompt
                section_text = simple_summarize(text)
                summaries[section] = f"{prompt}\n\n{section_text}"
        
        return summaries
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return {"Error": "Analysis failed. Please try again."}

# === PDF Export ===
def sanitize(text):
    """Clean text for PDF generation by replacing problematic characters."""
    # Replace common Unicode characters with their ASCII equivalents
    replacements = {
        '\u2010': '-',  # Hyphen
        '\u2011': '-',  # Non-breaking hyphen
        '\u2012': '-',  # Figure dash
        '\u2013': '-',  # En dash
        '\u2014': '--',  # Em dash
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2026': '...',  # Ellipsis
        '\u2212': '-',  # Minus sign
        '\u00b0': ' degrees',  # Degree symbol
        '\u00b5': 'u',  # Micro symbol
        '\u2122': '(TM)',  # Trademark symbol
        '\u00ae': '(R)',  # Registered trademark symbol
        '\u00a9': '(C)',  # Copyright symbol
        '\u00b1': '+/-',  # Plus-minus symbol
        '\u2264': '<=',  # Less than or equal to
        '\u2265': '>=',  # Greater than or equal to
        '\u2248': '~=',  # Approximately equal to
        '\u221e': 'infinity',  # Infinity symbol
        '\u03b1': 'alpha',  # Greek alpha
        '\u03b2': 'beta',  # Greek beta
        '\u03b3': 'gamma',  # Greek gamma
        '\u03b4': 'delta',  # Greek delta
        '\u03b5': 'epsilon',  # Greek epsilon
        '\u03b8': 'theta',  # Greek theta
        '\u03bb': 'lambda',  # Greek lambda
        '\u03bc': 'mu',  # Greek mu
        '\u03c0': 'pi',  # Greek pi
        '\u03c3': 'sigma',  # Greek sigma
        '\u03c9': 'omega',  # Greek omega
    }
    
    # Replace Unicode characters
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Remove any remaining non-ASCII characters
    text = ''.join(char for char in text if ord(char) < 128)
    
    return text

def create_pdf_report(filename, summaries, critique):
    """Create a PDF report with proper Unicode handling."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Set font to Arial which has better Unicode support
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, sanitize(f"BioSynthAI Analysis Report: {filename}"), ln=True)
    
    pdf.set_font("Arial", size=12)
    for section, content in summaries.items():
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, sanitize(f"{section} Summary"), ln=True)
        pdf.set_font("Arial", size=11)
        
        # Clean and wrap the content
        clean_content = sanitize(content)
        # Split content into paragraphs
        paragraphs = clean_content.split('\n\n')
        for para in paragraphs:
            # Split long paragraphs into lines that fit the page width
            words = para.split()
            line = []
            for word in words:
                if pdf.get_string_width(' '.join(line + [word])) < pdf.w - 20:
                    line.append(word)
                else:
                    pdf.multi_cell(0, 7, ' '.join(line))
                    line = [word]
            if line:
                pdf.multi_cell(0, 7, ' '.join(line))
            pdf.ln(5)  # Add space between paragraphs
    
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, sanitize("Methodological Critique"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, sanitize("See main analysis."))
    
    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    out_path = f"data/{filename}_analysis.pdf"
    
    try:
        pdf.output(out_path)
        return out_path
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def check_api_status():
    """Check OpenAI API status and return appropriate message."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "no_key", "No OpenAI API key found. The app will use local summarization."
    
    try:
        # Test API call
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}]
        )
        return "ok", None
    except Exception as e:
        if "insufficient_quota" in str(e):
            return "quota_exceeded", """
            OpenAI API quota exceeded. The app will use local summarization.
            
            To resolve this:
            1. Check your billing details at https://platform.openai.com/account/billing
            2. Add payment method or increase quota
            3. Or continue using the local summarization feature
            """
        return "error", f"OpenAI API error: {str(e)}. Using local summarization."

def create_concept_map(papers_info, section_summaries):
    dot = Digraph(comment='Concept Map', format='png')
    dot.attr(rankdir='LR', size='8,5')

    # Add paper nodes
    for paper in papers_info:
        dot.node(paper['title'], shape='box', style='filled', color='lightblue')

        # Add concept nodes and connect to paper
        for section, summary in section_summaries.get(paper['title'], {}).items():
            concept_node = f"{paper['title']}_{section}"
            dot.node(concept_node, label=section.title(), shape='ellipse', style='filled', color='lightgreen')
            dot.edge(paper['title'], concept_node, label='has section')

    # Optionally, add relationships between concepts (e.g., similar methods)
    # dot.edge('Paper1_methods', 'Paper2_methods', label='similar method')

    return dot

def analyze_concept_map(papers_info, section_summaries):
    """Generate a detailed analysis of the concept map."""
    try:
        # Count papers and sections
        num_papers = len(papers_info)
        sections_per_paper = {paper['title']: len(section_summaries.get(paper['title'], {})) 
                            for paper in papers_info}
        
        # Analyze section coverage
        all_sections = set()
        for sections in section_summaries.values():
            all_sections.update(sections.keys())
        
        # Analyze temporal distribution
        years = [paper.get('year', 'Unknown') for paper in papers_info]
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1
        
        # Find common sections across papers
        section_occurrences = {}
        for sections in section_summaries.values():
            for section in sections:
                section_occurrences[section] = section_occurrences.get(section, 0) + 1
        
        # Generate summary
        summary = f"""
### 📊 Concept Map Analysis

#### Overview
- Total Papers: {num_papers}
- Unique Sections: {len(all_sections)}
- Average Sections per Paper: {sum(sections_per_paper.values()) / num_papers:.1f}

#### Temporal Distribution
"""
        # Add temporal analysis
        for year, count in sorted(year_counts.items(), key=lambda x: x[0] if x[0] != 'Unknown' else '9999'):
            summary += f"- {year}: {count} paper{'s' if count > 1 else ''}\n"
        
        summary += "\n#### Section Analysis\n"
        # Add section coverage details with more context
        for section in sorted(all_sections):
            papers_with_section = section_occurrences[section]
            coverage = (papers_with_section / num_papers) * 100
            summary += f"- **{section.title()}**\n"
            summary += f"  - Coverage: {papers_with_section} papers ({coverage:.1f}%)\n"
            # Add papers that have this section
            papers_with_section_list = [p['title'] for p in papers_info 
                                      if section in section_summaries.get(p['title'], {})]
            summary += f"  - Found in: {', '.join(papers_with_section_list)}\n"
        
        summary += "\n#### Paper-Specific Analysis\n"
        # Add detailed paper insights
        for paper in papers_info:
            title = paper['title']
            sections = section_summaries.get(title, {})
            summary += f"- **{title}**\n"
            summary += f"  - Year: {paper.get('year', 'Unknown')}\n"
            summary += f"  - Authors: {paper.get('authors', 'Unknown')}\n"
            summary += f"  - Sections ({len(sections)}): {', '.join(sections.keys())}\n"
            # Add section summaries
            for section, content in sections.items():
                # Get first sentence of summary for each section
                first_sentence = content.split('.')[0] + '.'
                summary += f"    - {section.title()}: {first_sentence}\n"
        
        # Add relationship insights
        summary += "\n#### Key Relationships\n"
        # Find papers with similar sections
        for section in all_sections:
            papers_with_section = [p['title'] for p in papers_info 
                                 if section in section_summaries.get(p['title'], {})]
            if len(papers_with_section) > 1:
                summary += f"- Papers discussing {section.title()}: {', '.join(papers_with_section)}\n"
        
        return summary
    except Exception as e:
        return f"Error analyzing concept map: {str(e)}"

def process_chat_response(user_q, relevant_docs, papers_info, section_summaries):
    """Process chat responses for multi-document analysis."""
    try:
        # Initialize response components
        response = ""
        used_sources = set()
        
        # Group papers by their content for analysis
        papers_by_section = {}
        
        # Process each document
        for doc in relevant_docs:
            source = doc.metadata["source"]
            used_sources.add(source)
            
            # Get the paper info
            paper_info = next((p for p in papers_info if p["title"] == source), None)
            
            if paper_info:
                # Initialize paper entry if not exists
                if source not in papers_by_section:
                    papers_by_section[source] = {
                        "metadata": paper_info,
                        "sections": {},
                        "full_text": doc.page_content
                    }
                
                # Get section summaries
                if source in section_summaries:
                    for section, summary in section_summaries[source].items():
                        papers_by_section[source]["sections"][section] = summary
        
        # Generate response
        if papers_by_section:
            response = "Here's the analysis based on your question:\n\n"
            
            # Add overview of papers being analyzed
            response += "**Papers Analyzed:**\n"
            for source in papers_by_section:
                paper_info = papers_by_section[source]["metadata"]
                response += f"- {source} ({paper_info.get('year', 'Unknown')})\n"
            response += "\n"
            
            # Add content from each paper
            for source, paper_data in papers_by_section.items():
                response += f"\n**From {source}:**\n"
                
                # Add section summaries if available
                if paper_data["sections"]:
                    for section, summary in paper_data["sections"].items():
                        response += f"\n{section.title()}:\n{summary}\n"
                else:
                    # If no section summaries, use the full text
                    response += f"\nContent:\n{paper_data['full_text'][:500]}...\n"
            
            # Add synthesis if multiple papers
            if len(papers_by_section) > 1:
                response += "\n**Overall Synthesis:**\n"
                for source, paper_data in papers_by_section.items():
                    response += f"\n{source}:\n"
                    if paper_data["sections"]:
                        for section, summary in paper_data["sections"].items():
                            response += f"- {section.title()}: {summary.split('.')[0]}.\n"
                    else:
                        response += f"- {paper_data['full_text'][:200]}...\n"
        else:
            # If no papers found in papers_by_section, try to get content directly from relevant_docs
            response = "Here's what I found in the papers:\n\n"
            for doc in relevant_docs:
                source = doc.metadata["source"]
                response += f"\n**From {source}:**\n"
                response += f"{doc.page_content[:500]}...\n"
            
            if not response:
                response = "I couldn't find specific information related to your question in the uploaded papers. Please try rephrasing your question or check if the relevant papers are uploaded."
        
        return response
        
    except Exception as e:
        return f"Error processing response: {str(e)}"

# === Streamlit UI ===
st.set_page_config(page_title="BioSynthAI Analyzer", layout="wide")

# Initialize session states
if "single_doc_analysis" not in st.session_state:
    st.session_state.single_doc_analysis = {
        "uploaded_file": None,
        "analysis_complete": False,
        "summaries": None
    }

if "kb_built" not in st.session_state:
    st.session_state.kb_built = False

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "papers_info" not in st.session_state:
    st.session_state.papers_info = []

if "section_summaries" not in st.session_state:
    st.session_state.section_summaries = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header with API status
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🧠 BioSynthAI – Academic PDF Assistant")
with col2:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.caption("🔴 No API key")
    else:
        try:
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}]
            )
            st.caption("🟢 API Connected")
        except Exception as e:
            if "insufficient_quota" in str(e):
                st.caption("")
            else:
                st.caption("🟡 API Error")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Choose a mode:", ["Single Document Analysis", "Multi-Document Chat"])

# Main content area
if mode == "Single Document Analysis":
    # Add reset button at the top
    if st.session_state.single_doc_analysis["analysis_complete"]:
        if st.button("🔄 Reset Analysis"):
            # Clear the session state
            st.session_state.single_doc_analysis = {
                "uploaded_file": None,
                "analysis_complete": False,
                "summaries": None
            }
            # Clear any uploaded files
            upload_dir = os.path.join("data", "uploaded")
            for file in os.listdir(upload_dir):
                if file.endswith(".pdf"):
                    os.remove(os.path.join(upload_dir, file))
            st.rerun()

    # Only show uploader if no analysis is complete
    if not st.session_state.single_doc_analysis["analysis_complete"]:
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file:
            st.session_state.single_doc_analysis["uploaded_file"] = uploaded_file
            if st.button("Analyze Paper"):
                with st.spinner("Analyzing document..."):
                    # Ensure upload directory exists
                    upload_dir = os.path.join("data", "uploaded")
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    # Save the uploaded file
                    file_path = os.path.join(upload_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract text and run analysis
                    text = extract_text_from_pdf(file_path)
                    summaries = run_deep_analysis(text)
                    st.session_state.single_doc_analysis["summaries"] = summaries
                    st.session_state.single_doc_analysis["analysis_complete"] = True
                    st.rerun()
    else:
        # Display the analysis results
        st.success("Analysis complete!")
        for sec, content in st.session_state.single_doc_analysis["summaries"].items():
            st.subheader(sec)
            st.write(content)
        
        # Add download button for PDF report
        if st.session_state.single_doc_analysis["summaries"]:
            report_path = create_pdf_report(
                st.session_state.single_doc_analysis["uploaded_file"].name.replace(".pdf", ""),
                st.session_state.single_doc_analysis["summaries"],
                ""
            )
            if report_path:
                with open(report_path, "rb") as f:
                    st.download_button(
                        "📥 Download PDF Report",
                        data=f,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf"
                    )

# === Multi-Document Chat Mode ===
elif mode == "Multi-Document Chat":
    st.info("Upload multiple PDFs, then ask questions that compare or explore them.")
    
    # Ensure upload directory exists
    upload_dir = os.path.join("data", "uploaded")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Only show uploader and build button if knowledge base is not built
    if not st.session_state.kb_built:
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
        if st.button("Build Knowledge Base"):
            if "uploaded_files" not in st.session_state or not st.session_state.uploaded_files:
                st.error("Please upload at least one PDF before building the knowledge base.")
                st.stop()
            
            # Clear previous analysis
            st.session_state.retriever = None
            st.session_state.papers_info = []
            st.session_state.section_summaries = {}
            st.session_state.chat_history = []
            
            # Remove uploaded files from previous runs
            for file in os.listdir(upload_dir):
                if file.endswith(".pdf"):
                    os.remove(os.path.join(upload_dir, file))
            
            # Save new files
            for f in st.session_state.uploaded_files:
                file_path = os.path.join(upload_dir, f.name)
                with open(file_path, "wb") as out:
                    out.write(f.getbuffer())
                st.write(f"Saved: {f.name}")
            
            with st.spinner("Indexing all documents..."):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=50,
                    length_function=len,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                )
                docs, metas = [], []
                papers_info = []
                section_summaries = {}
                
                # Get list of files in upload directory
                files_to_process = [f for f in os.listdir(upload_dir) if f.endswith('.pdf')]
                
                if not files_to_process:
                    st.error("No PDF files found in the upload directory. Please upload some files first.")
                    st.stop()
                
                # Process files
                for file_name in files_to_process:
                    file_path = os.path.join(upload_dir, file_name)
                    try:
                        text = extract_text_from_pdf(file_path)
                        chunks = splitter.split_text(text)
                        docs.extend(chunks)
                        metas.extend([{ "source": file_name }] * len(chunks))
                        paper_info = extract_sections_and_metadata(file_path)
                        papers_info.append(paper_info)
                        section_summaries[paper_info["title"]] = {}
                        for sec, text in paper_info["sections"].items():
                            section_summaries[paper_info["title"]][sec] = summarize_section(text, sec, openai_api_key=os.getenv("OPENAI_API_KEY"))
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {str(e)}")
                        continue
                
                if not docs:
                    st.error("No documents were successfully processed. Please check your PDF files and try again.")
                    st.stop()
                
                # Create vector store
                vector_store = FAISS.from_texts(docs, embedding=embeddings, metadatas=metas)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                
                # Store the retriever in session state
                st.session_state.retriever = retriever
                st.session_state.papers_info = papers_info
                st.session_state.section_summaries = section_summaries
                st.session_state.kb_built = True
                st.rerun()

    # Only show chat/concept map if knowledge base is built
    if st.session_state.kb_built:
        st.markdown("## 📊 Paper Concept Map (Professional Diagram)")
        try:
            st.graphviz_chart(create_concept_map(st.session_state.papers_info, st.session_state.section_summaries))
            st.markdown(analyze_concept_map(st.session_state.papers_info, st.session_state.section_summaries))
        except Exception as e:
            st.error(f"Error in concept map generation: {str(e)}")
            st.info("Please try rebuilding the knowledge base.")

        # Chat interface
        if "retriever" in st.session_state:
            # Chat input using a form
            with st.form(key='chat_form', clear_on_submit=True):
                user_q = st.text_input(
                    "Ask questions about the papers' content, methods, results, conclusions, or citation details."
                )
                submit_button = st.form_submit_button(label='Send')
            
            if submit_button and user_q:
                # Get relevant documents
                retriever = st.session_state.retriever
                retriever.search_kwargs["k"] = 5
                relevant_docs = retriever.get_relevant_documents(user_q)
                
                # Process and display response
                response = process_chat_response(user_q, relevant_docs, st.session_state.papers_info, st.session_state.section_summaries)
                st.session_state.chat_history.append({"user": user_q, "assistant": response})
                
                # Display chat history
                for chat in st.session_state.chat_history:
                    st.write("**You:**", chat["user"])
                    st.write("**Assistant:**", chat["assistant"])
                    st.markdown("---")
            
            # Add restart button at the bottom
            if st.button("🔄 Restart Analysis"):
                # Clear all session state
                for key in ["retriever", "papers_info", "section_summaries", "chat_history", "kb_built", "uploaded_files"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # Clear uploaded files
                for file in os.listdir(upload_dir):
                    if file.endswith(".pdf"):
                        os.remove(os.path.join(upload_dir, file))
                st.rerun()
    else:
        # Show upload and build UI
        st.info("Please build the knowledge base first.")
