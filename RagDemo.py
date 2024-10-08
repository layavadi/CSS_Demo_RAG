
from transformers import AutoTokenizer, AutoModel
import torch
from opensearchpy import OpenSearch
import json
import numpy as np
import os
import PyPDF2
import gradio as gr
import re
from flask import Flask, send_file, abort,request

#from googletrans import Translator

#translator = Translator()
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
model = AutoModel.from_pretrained('bert-base-multilingual-cased')
folder_path = os.getenv("DOC_PATH",'/Users/vbhatt/opensearch/Demo')

def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Average pooling
    return embeddings[0].numpy()

# Step 3: Perform vector search in OpenSearch
def search_by_vector(query_vector, opensearch_client, index_name, top_k=5):
    query = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector.tolist(),
                    "k": top_k
                }
            }
        },
        "_source": ["text"]
    }
    response = opensearch_client.search(index=index_name, body=query)
    # Print the number of hits
    number_of_hits = response['hits']['total']['value']
    print(f"Number of hits: {number_of_hits}")

    # Print the IDs and scores of the hits
    print("Hit IDs and Scores:")
     
    hits = response['hits']['hits']
    contexts = []
    for hit in hits:
        doc_id = hit['_id']  # Document ID containing name and chunk info
        doc_name, chunk = doc_id.split('_chunk_')  # Assuming ID is formatted as 'docname_chunkN'
        context = hit['_source'].get('text', 'No context available')
        score = hit['_score']
        print(f"ID: {doc_id}, chunk: {chunk} Score: {score}")
        
        contexts.append({
            'document': doc_name,
            'chunk': chunk,
            'context': context[:500],  # Snippet of the first 500 characters
            'score': score
        })
    return contexts
    #return response['hits']['hits']

# Step 4: Send retrieved context to LLM for final answer
def query_llm(query, context):
    # Example with OpenAI GPT API
    from openai import AzureOpenAI
    azure_openai_api_version = "2024-03-01-preview"
    azure_openai_endpoint = "https://cx-genius-dev.openai.azure.com/"
    azure_openai_deployment = "gpt-35-turbo-16k"

    client = AzureOpenAI(
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        api_version=azure_openai_api_version,
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint=azure_openai_endpoint,
    )

    completion = client.chat.completions.create(
        model=azure_openai_deployment,  # e.g. gpt4-turbo
        messages=[
            {
                "role": "user",
                 "content": "צור את התגובה לשאלה זו: `" + query + "` בהתבסס על התוכן שסופק להלן: `" + context + "`",
            },
        ],
    )
    return (completion.choices[0].message.content)

def create_opensearch_client(host, port, user, password):
    try:
        # Connect to OpenSearch
        client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=(user, password),
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False
        )
        print("Connected to OpenSearch!")
        return client
    except Exception as e:
        print(f"Error connecting to OpenSearch: {e}")
        return None
    

def create_index_with_vector_field(client, index_name, dimensions):
    if client.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists.")
    else:
        index_body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn": True  # Enable k-Nearest Neighbors for nmslib
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "english_translation": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",  # Vector type field
                        "dimension": dimensions,  # Number of dimensions from the embedding model
                        "method": {
                            "name": "hnsw",  # Method for the vector search
                            "space_type": "l2",  # Euclidean distance for similarity
                            "engine": "nmslib"  # Use nmslib as the vector search engine
                        }
                    }
                }
            }
        }
        client.indices.create(index=index_name, body=index_body)
        print(f"Index '{index_name}' created successfully.")

# Step 3: Insert Embeddings and Text into OpenSearch
def insert_document(client, index_name, doc_id, text, embedding):

    # Prepare document with both Hebrew and English texts
    english_translation = translate_to_english(text)
    document = {
        "text": text,
        "english_translation": english_translation,
        "embedding": embedding.tolist()  # Convert numpy array to list before indexing
    }
    response = client.index(index=index_name, id=doc_id, body=document)
    return response

# Function to read and chunk a PDF into text chunks
def chunk_pdf(pdf_path, chunk_size=500):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    text = re.sub(r'[\n\t\r]', ' ', text)  # Replace newlines, tabs, and carriage returns with a space
    text = re.sub(r' +', ' ', text)  # Replace multiple spaces with a single space

    # Split text into chunks of specified size (e.g., 500 characters)
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

def translate_to_english(hebrew_text):
    translated = translator.translate(str(hebrew_text), src='he', dest='en')
    return translated.text

# Function to process all PDF files in a folder and insert into OpenSearch
def process_pdf_folder_and_insert_to_opensearch(folder_path, client, index_name):
    # Iterate over all PDF files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)

            # Iterate through chunks of the PDF
            for i, chunk in enumerate(chunk_pdf(pdf_path)):
                embedding =  text_to_embedding(chunk)

                # Create a unique document ID using the file name and chunk index
                doc_id = f"{filename}_chunk_{i+1}"

                # Insert chunk and its embedding into OpenSearch
                insert_document(client, index_name, doc_id, chunk, embedding)

                print(f"Inserted chunk {i+1} of file {filename} into index {index_name}")

def handle_user_query(query, client, index_name):
    query_vector = text_to_embedding(query)
    contexts = search_by_vector(query_vector, client, index_name)

    if not contexts:
        return "No relevant information found."

    # Combine the text from the retrieved results
   # context = "\n".join([result['_source']['text'] for result in results])
    context_text = " ".join([ctx['context'] for ctx in contexts])  # Join the retrieved contexts
    context_text = context_text[:1000]
    #answer = query_llm(query, context_text)
    answer = 'הכללים ליצוא אתרוגים בהתבסס על התוכן שסופק הם:\n1. האתרוגים נארזים ליצוא לצורך בדיקתם וכשירותם לייצוא לחו"ל.\n2. האתרוגים שנקטפו לצרכי Citrus medica הם פירות של עץ אתרוגים.\n3. בזמן הייבוש של האתרוג, יש לדת - בליטה בראש האתרוג במקום שבו נשר הפרח, ונחשב פיטם.\n4. הסימן הידור צריך להיות לסמל האתרוג.\n5. מחלת הדרים הנגרמת ע"י הפטריה מלסקו (קימלון) - הנפוצה מאוד בעצי אתרוגים ומהווה מחלת tracheiphila הסגר במספר מדינות לפחות 5 ימים.\n6. סגירת האתרוגים למשך הסגר במטרה להבליט סימני מזיקים כמו זבוב הים התיכון.\n7. התשמיד יש לבדוק רווח אמצעי מהזנב הימני של התוצר (י׀יך תאריך החתימה ודפ ה-\'עמוד מס 2 נספח טופס בדיקה ואישור אתר לביקורת אתרוגים ליצוא\'.)\n8. האתרוגים יבדקו לפני ביקורת במס\' 50 מרחק עד הוראת עבודה ליצוא אתרוגים 11.07.2018.\n9. האריזה של האתרוגים תוצרת ממוספרת ומסומנת, ותהיה בהתאם להוראת עבודה מס 22, גרסה מס 00-02-25.\n10. האתרים נבדקים לניקיון ולסביבתם הכוללת מבנה מחופה בגג קשיח.'


    # doc_display = ""
    # for i, ctx in enumerate(contexts):
    #     doc_display += f"**Document {i+1}: {ctx['document']}** (Chunk: {ctx['chunk']})\n\n"
    #     doc_display += f"Context: {ctx['context']}\n\n"
    #     doc_display += f"Score: {ctx['score']}\n\n"
    #     doc_display += "---\n\n"
    return answer,contexts

def format_results(results):
    # HTML table structure
    table = """
    <table border="1" style="width:100%; text-align: left;">
      <tr>
        <th>Document Name</th>
        <th>Context</th>
        <th>Score</th>
      </tr>
    """
    host  = "http://127.0.0.1:5000"
    # Add rows to the table for each result
    for result in results:
        # Extract document name and chunk from doc_id
        doc_name = result['document']
        chunk = result['chunk']
        doc_url = f"{host}/get-pdf/{doc_name}"
        
        # Add the table row for each document
        table += f"""
        <tr>
          <td><a href="javascript:void(0);" onclick="window.open('{doc_url}', 'popup', 'width=800,height=600');">{doc_name} (Chunk: {chunk})</a></td>
          <td><div style="direction: rtl; text-align: right;">{result['context']}</div></td>
          <td>{result['score']}</td>
        </tr>
        """
    table += "</table>"
    
    return table


# Gradio function to be triggered from the interface
def gradio_function(query):

    host = os.getenv('CSS_HOST','localhost')
    port = os.getenv('CSS_PORT',9200)
    username = os.getenv('CSS_USER','admin')
    password = os.getenv('CSS_PASSWORD','admin')
    index_name = 'hebrew_docs_index'

    # Connect to OpenSearch
    client = create_opensearch_client(host, port, username, password)

    # Process the user query and return the LLM answer
    answer, relevant_documents = handle_user_query(query, client, index_name)
    
    document_table = format_results(relevant_documents)
    return answer, document_table
   # return answer, document_table

def create_gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("### Hebrew Query Search with LLM")

        query_input = gr.Textbox(label="Enter your query in Hebrew")
        output_text = gr.Textbox(label="LLM Response")
        context_output = gr.HTML(label="Relevant Documents and Contexts")

        query_button = gr.Button("Submit")

       # query_button.click(fn=gradio_function, inputs=query_input, outputs=output_text)
        query_button.click(
            fn=gradio_function,  # Function to call
            inputs=[query_input],  # Inputs to the function
            outputs=[output_text, context_output]  # Outputs to display
        )

    return demo 

app = Flask(__name__)

# Route to serve the PDF file securely
@app.route('/get-pdf/<doc_name>')
def get_pdf(doc_name):
    # Assuming your PDF files are stored in a secure directory on the server
    pdf_path = os.path.join(folder_path, doc_name)
    
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=False)  # Sends the PDF to the frontend
    else:
        abort(404)  # Return a 404 if the file is not found

def run_flask():
    app.run(port=5000) 

if __name__ == "__main__":


    host = os.getenv('CSS_HOST','localhost')
    port = os.getenv('CSS_PORT',9200)
    username = os.getenv('CSS_USER','admin')
    password = os.getenv('CSS_PASSWORD','admin')
    index_name = 'hebrew_docs_index'
    dimensions = 768  # Based on your embedding model

    # Connect to OpenSearch
    client = create_opensearch_client(host, port, username, password)

    # Create index for vector search with nmslib engine
    #create_index_with_vector_field(client, index_name, dimensions)

    # Process folder and insert documents into OpenSearch
    #process_pdf_folder_and_insert_to_opensearch(folder_path, client, index_name)

    # Convert user query to vector and search in OpenSearch
    # query = "מהם הכללים ליצוא אתרוגים"
    # query_vector = text_to_embedding(query)
    # results = search_by_vector(query_vector, client, index_name)

    # # Combine retrieved text and send to LLM
    # context = "\n".join([result['_source']['text'] for result in results])
    # context = context[:1000]
    # answer = query_llm(query,context)
    # print("Answer:", answer)
    import threading
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    gradio_app = create_gradio_ui()
    gradio_app.launch()