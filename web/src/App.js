import React, { useState } from 'react';
import axios from 'axios';
import Chat, { Bubble, useMessages } from '@chatui/core';
import '@chatui/core/dist/index.css';

const App = () => {
  const { messages, appendMsg, setTyping } = useMessages([]);

  async function handleSend(type, val) {
    if (type === 'text' && val.trim()) {
      appendMsg({
        type: 'text',
        content: { text: val },
        position: 'right',
      });

      setTyping(true);

      const response = await axios.post('http://127.0.0.1:5000/chat', {
        user_id: 'buryrocks',
        query: val,
      });

      appendMsg({
        type: 'text',
        content: { text: response.data.response },
        position: 'left',
      });

      setTyping(false);
    }
  }

  function renderMessageContent(msg) {
    const { content } = msg;
    return <Bubble content={content.text} />;
  }

  return (
      <Chat
          navbar={{ title: 'Assistant' }}
          messages={messages}
          renderMessageContent={renderMessageContent}
          onSend={handleSend}
      />
  );
};

export default App;
