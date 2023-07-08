import React, { useState } from 'react';
import axios from 'axios';
import Cookies from 'js-cookie';
import Chat, { Bubble, useMessages, Input, Button, Text } from '@chatui/core';
import '@chatui/core/dist/index.css';

const COOKIE_KEY = 'my_diary_with_bible_username'

const Form = ({
  saveUsername
}) => {
  const [username, setUsername] = useState('');
  return (
    <div>
      <h3>Welcome to the Bible Verse Chat!</h3>
      <Input maxLength={20} value={username} onChange={val => setUsername(val)} placeholder="Please type your user name..." />
      <Text breakWord>This is a hackathon project, username is stored in the cookies, please set a secret name and keep it private. </Text>
      <Button color="primary" onClick={() => {
        Cookies.set(COOKIE_KEY, username)
        saveUsername(username)
      }}>Save</Button>
    </div>
  );
}

const App = () => {
  const [username, setUsername] = useState(Cookies.get(COOKIE_KEY));
  const hasUsername = !!username
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
        user_id: username,
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
    hasUsername ? 
    <Chat
      navbar={{ title: 'Assistant' }}
      messages={messages}
      renderMessageContent={renderMessageContent}
      onSend={handleSend}
    /> : 
    <Form saveUsername={setUsername}/>
  );
};

export default App;
