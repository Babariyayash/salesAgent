from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

from salesgpt.salesgpt_agent import SalesGPT
from config import OPENAI_API_KEY


class LanguageModelProcessor:
    def __init__(self):
        self.sales_agent = self.initialize_sales_gpt()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load the system prompt from a file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.sales_agent.sales_conversation_utterance_chain.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def initialize_sales_gpt(self):
        config = dict(
            salesperson_name="June",
            salesperson_role="Sales Representative",
            company_name="Eco Home Goods",
            company_business="Eco Home Goods offers a variety of eco-friendly household products that are both sustainable and affordable. We believe in providing high-quality products that help our customers live a more environmentally friendly lifestyle.",
            company_values="Our mission is to promote sustainable living by offering products that are eco-friendly and affordable. We are committed to reducing our carbon footprint and helping our customers do the same.",
            conversation_purpose="find out if they are interested in our new line of eco-friendly kitchen products.",
            conversation_history=[],
            conversation_type="call",
            use_tools=False,  # Assuming tools are not required in this integration
            product_catalog="salesgpt/sample_product_catalog.txt"
        )

        llm = ChatOpenAI(model='gpt-4-turbo', temperature=0.9, openai_api_key=OPENAI_API_KEY)
        sales_agent = SalesGPT.from_llm(llm, verbose=False, **config)
        sales_agent.seed_agent()
        return sales_agent

    def start_conversation(self):
        # The initial message is generated when the agent is seeded
        initial_response = self.sales_agent.conversation_history[-1].split(":")[1].strip(" <END_OF_TURN>")

        self.memory.chat_memory.add_ai_message(initial_response)  # Add AI response to memory

        return initial_response

    def process(self, text):
        self.sales_agent.human_step(text)
        self.sales_agent.determine_conversation_stage()
        self.sales_agent.step()

        # Get the latest response from SalesGPT
        latest_response = self.sales_agent.conversation_history[-1].split(":")[1].strip(" <END_OF_TURN>")

        self.memory.chat_memory.add_user_message(text)  # Add user message to memory
        self.memory.chat_memory.add_ai_message(latest_response)  # Add AI response to memory

        return latest_response
