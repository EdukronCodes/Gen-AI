import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import ChatWindow from "@/components/chat/chat-window";
import { User } from "@shared/schema";

export default function Chat() {
  const [currentUserId] = useState(1); // Default user for demo

  const { data: user, isLoading: userLoading } = useQuery({
    queryKey: ["/api/users", currentUserId],
    enabled: !!currentUserId,
  });

  const { data: chatHistory, isLoading: chatLoading } = useQuery({
    queryKey: ["/api/chat/history", currentUserId],
    enabled: !!currentUserId,
  });

  if (userLoading || chatLoading) {
    return (
      <div className="flex h-screen bg-gray-50 items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading RetailBot...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <ChatWindow user={user as User} chatHistory={chatHistory || []} />
    </div>
  );
}
