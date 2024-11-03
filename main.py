import asyncio
from conversation_manager import ConversationManager

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
