
import click
from langchain_core.prompts import PromptTemplate # type: ignore
from model import LLmModel, LangChainLLMWrapper
from vectore_store import VectoreStore
import model_config as conf
from print_exam import print_exam

def generate_exam_response(retriver, llm, topics=None, num_questions=5, level="easy"):
    prompt = PromptTemplate(
        template=conf.user_template, 
        input_variables=["num_questions", "context", "level"]
    )

    if topics:
        search_query = ", ".join(topics)
    else:
        search_query = "main concept 'not important actually'"
    
    retrived_docs = retriver.invoke(search_query)
    context_text = "\n\n".join([doc.page_content for doc in retrived_docs])
    formatted_user_request = prompt.format(num_questions=num_questions, context=context_text, level=level)
    final_prompt = conf.system_prompt + "\n\n" + formatted_user_request
    response = llm.invoke(final_prompt)

    # context is generated automatically from ConversationalRetrievalChain, so we only need to pass the question
    # response = agent_chain.invoke({"question": formatted_query})
    return response

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--num_questions', '-n', default=4, help='Number of questions to generate')
@click.option('--pages', '-p', help='choose the number of pages to generate exam from, separated by comma')
@click.option('--topics', '-t', help='choose the topics to generate exam from, separated by comma')
@click.option('--level', '-l', type=click.Choice(['easy', 'medium', 'hard']), default='easy', help='Difficulty level of the questions')
def main(file_path, num_questions, pages, topics, level):
    if pages:
        print('pages are :\n\n', pages)
        content = [int(str(p).strip()) - 1 for p in pages.strip().split(',')]
        print('after pages are :\n\n', pages)
        type = "pages"
    else:
        print('topics are :\n\n', topics)
        content = topics.strip().split(',')
        print(' after topics are :\n\n', topics)
        type = "topics"
   
    print("Loading PDF and preparing data...")
    
    base_model = LLmModel()
    llm = LangChainLLMWrapper(base_model)

    print("Creating retriever and LLM...")
   
    if type == "pages":
        ret = VectoreStore(file_path=file_path, filter_value=content, filter_type="pages")
        
    else:
        ret = VectoreStore(file_path=file_path, filter_value=content, filter_type="topics")

    retriver = ret()

    if type == "pages":
        response = generate_exam_response(retriver, llm, num_questions=num_questions, level=level)
    else:
        response = generate_exam_response(retriver, llm, topics=content, num_questions=num_questions, level=level)

    # print("response is:\n\n",response)

    print_exam(response)

if __name__ == "__main__":
    main()