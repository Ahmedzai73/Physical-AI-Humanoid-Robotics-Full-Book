import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import RAGChatbot from '@site/src/components/RAGChatbot';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
        <RAGChatbot />
      </OriginalLayout>
    </>
  );
}