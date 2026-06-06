import streamlit as st
import requests

BACKEND_URL = 'http://localhost:8000'
n_results = 3

st.set_page_config(
    page_title="Chat with your pdfs",
    page_icon="",
    layout="wide"
)

st.title('Chat with your PDFS')
st.caption('Powered by RAG (Retrieval Augmented Generation)')

with st.sidebar:
    st.header('Your documents')
    uploaded_file = st.file_uploader(
        'Upload a PDF',
        type=['pdf'],
        help='Upload any pdf and start asking questions'
    )

    if uploaded_file:
        if st.button('Ingest PDF', type='primary', use_container_width=True):
            with st.spinner('Reading, chunking and embedding your pdf'):
                response = requests.post(
                    f'{BACKEND_URL}/ingest',
                    files={'file': (uploaded_file.name, uploaded_file, 'application/pdf')}
                )

            if response.status_code == 200:
                data = response.json()
                st.success(f"Done. Added {data['chunks_added']} chunks")
                st.info(f"Total in database: {data['total_chunks']} chunks")
            else:
                st.error('Something went wrong')

    st.divider()

    try:
        status = requests.get(f'{BACKEND_URL}/').json()
        st.metric('Chunks in database', status['total_chunks'])
    except Exception as e:
        print(e)
        st.error(f'Something went wrong {str(e)}')

    st.divider()

    # Document selector
    try:
        docs_response = requests.get(f'{BACKEND_URL}/documents').json()
        sources = docs_response.get('sources', [])
    except Exception as e:
        print(e)
        sources = []

    if sources:
        selected_source = st.selectbox(
            'Search in document',
            options=['All documents'] + sorted(sources),
        )
    else:
        selected_source = 'All documents'
        st.caption('No documents ingested yet.')

    st.divider()

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

    if msg['role'] == 'assistant' and 'sources' in msg:
        with st.expander('View retrieved chunks', expanded=False):
            for source in msg['sources']:
                st.markdown(
                    f"**{source['source']} - Page {source['page']}** "
                    f"*(Similarity score: {source['score']:.3f})*"
                )

if question := st.chat_input('Ask anything about your documents'):
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.write(question)

    with st.chat_message('assistant'):
        with st.spinner('Searching your documents...'):
            try:
                source_filter = None if selected_source == 'All documents' else selected_source
                response = requests.post(
                    f'{BACKEND_URL}/ask',
                    json={
                        'question': question,
                        'n_results': n_results,
                        'source': source_filter
                    }
                )
                data = response.json()
                if 'error' in data:
                    answer = f"{data['error']}"
                    sources = []
                else:
                    answer = data['answer']
                    sources = data['sources']
            except Exception as e:
                print(e)
                answer = f"Something went wrong {str(e)}"
                sources = []

        st.write(answer)

        if sources:
            with st.expander('View retrieved chunks', expanded=False):
                for source in sources:
                    st.markdown(
                        f"**{source['source']} - Page {source['page']}** "
                        f"*(Similarity score: {source['score']:.3f})*"
                    )

    st.session_state.messages.append({
        'role': 'assistant',
        'content': answer,
        'sources': sources
    })