import { Bot } from "lucide-react";

interface MessageBubbleProps {
  message: string;
  type: "user" | "bot";
  timestamp: Date;
  userInitials: string;
}

export default function MessageBubble({ message, type, timestamp, userInitials }: MessageBubbleProps) {
  if (type === "user") {
    return (
      <div className="flex items-start space-x-3 justify-end">
        <div className="user-message-bg rounded-2xl rounded-tr-lg p-4 shadow-sm max-w-md">
          <p className="text-white">{message}</p>
        </div>
        <div className="w-8 h-8 bg-gradient-to-br from-gray-400 to-gray-600 rounded-full flex items-center justify-center flex-shrink-0">
          <span className="text-white text-sm font-medium">{userInitials}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start space-x-3">
      <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center flex-shrink-0">
        <Bot className="text-white text-sm" />
      </div>
      <div className="bot-message-bg rounded-2xl rounded-tl-lg p-4 shadow-sm max-w-2xl">
        <p className="text-gray-800">{message}</p>
      </div>
    </div>
  );
}
