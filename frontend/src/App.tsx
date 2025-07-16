import { useState } from 'react';
import type { FC, FormEvent } from 'react';

// Define the API endpoint as a constant
const API_URL = 'http://127.0.0.1:8080/infer';

/**
 * Defines the structure of a single chat message for the frontend UI.
 */
interface Message {
  role: 'user' | 'bot';
  text: string;
}

/**
 * Defines the expected JSON response structure from the inference API.
 */
interface ApiInferenceResponse {
  generated_text: string;
}

/**
 * Defines the structure for the API request payload.
 */
interface ApiInferenceRequest {
  messages: {
    role: 'user' | 'assistant' | 'system';
    content: string;
  }[];
}

/**
 * The main App component for the Rust-Llama Chat interface.
 */
const App: FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [prompt, setPrompt] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  /**
   * Handles the form submission event.
   * Sends the full conversation history to the API and updates the chat.
   */
  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    const userMessage: Message = { role: 'user', text: prompt };
    const updatedMessages = [...messages, userMessage];

    setMessages(updatedMessages);
    setLoading(true);
    setPrompt('');

    try {
      // Map the frontend message format to the backend API format.
      const apiPayload: ApiInferenceRequest = {
        messages: updatedMessages.map(msg => ({
          role: msg.role === 'bot' ? 'assistant' : msg.role,
          content: msg.text,
        })),
      };

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiPayload),
      });

      if (!response.ok) {
        throw new Error(`Network error: ${response.statusText}`);
      }

      const data = (await response.json()) as ApiInferenceResponse;

      const botMessage: Message = {
        role: 'bot',
        text: data.generated_text.trim(),
      };

      setMessages((prevMessages) => [...prevMessages, botMessage]);

    } catch (error) {
      console.error('Failed to contact server:', error);
      const errorMessage: Message = {
        role: 'bot',
        text: '❌ Error: Could not connect to the server.',
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <h1>Rust-Llama Chat</h1>

      <div className="chat-window">
        {messages.map((message, index) => (
          <div key={index} className={`msg ${message.role}`}>
            <span className="role">{message.role}:</span> {message.text}
          </div>
        ))}
        {loading && <div className="msg bot typing">Bot is typing…</div>}
      </div>

      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Type something…"
          disabled={loading}
          autoFocus
        />
        <button type="submit" disabled={loading}>
          Send
        </button>
      </form>
    </div>
  );
};

export default App;