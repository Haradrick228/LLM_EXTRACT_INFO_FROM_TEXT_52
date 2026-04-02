from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from common.interfaces.user_interaction import BotAdapter

def create_app(adapter: BotAdapter) -> FastAPI:
    """
    Factory for FastAPI app with chat endpoint.
    """
    app = FastAPI(
        title="Chatbot REST API",
        version="1.0.0",
        description="REST API for sending messages to any BotAdapter implementation"
    )

    class MessageRequest(BaseModel):
        message: str

    class MessageResponse(BaseModel):
        response: str

    @app.post(
        "/chat",
        response_model=MessageResponse,
        summary="Send a message to the bot"
    )
    async def chat_endpoint(req: MessageRequest) -> MessageResponse:
        """
        Принимает JSON {'message': 'text'}, передаёт в BotAdapter и возвращает ответ.
        """
        try:
            import inspect
            print(f"[DEBUG] adapter: {adapter!r}, type: {type(adapter)}")
            print(f"[DEBUG] handle_message sig: {inspect.signature(adapter.handle_message)}")

            bot_reply = await adapter.handle_message(req.message)
            return MessageResponse(response=bot_reply)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))



    return app