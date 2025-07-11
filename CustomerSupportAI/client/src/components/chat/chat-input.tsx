import { useState, useRef, useEffect } from "react";
import { Send, Paperclip, Mic, Truck, RotateCcw, Search, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  onQuickAction: (action: string) => void;
  isLoading: boolean;
}

export default function ChatInput({ onSendMessage, onQuickAction, isLoading }: ChatInputProps) {
  const [message, setMessage] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (message.trim() && !isLoading) {
      onSendMessage(message);
      setMessage("");
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }
  }, [message]);

  const quickActions = [
    { id: "track-order", label: "Track Order", icon: Truck },
    { id: "return-item", label: "Return Item", icon: RotateCcw },
    { id: "product-search", label: "Product Search", icon: Search },
    { id: "recommendations", label: "Recommendations", icon: Star },
  ];

  return (
    <div className="bg-white border-t border-gray-200 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Quick Action Buttons */}
        <div className="flex flex-wrap gap-2 mb-4">
          {quickActions.map((action) => {
            const Icon = action.icon;
            return (
              <Button
                key={action.id}
                variant="outline"
                size="sm"
                onClick={() => onQuickAction(action.id)}
                className="bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100"
              >
                <Icon className="w-4 h-4 mr-1" />
                {action.label}
              </Button>
            );
          })}
        </div>

        {/* Message Input */}
        <div className="flex items-end space-x-4">
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              className="min-h-[44px] max-h-32 resize-none pr-20 rounded-2xl border-gray-300 focus:ring-2 focus:ring-primary focus:border-transparent"
              rows={1}
              disabled={isLoading}
            />
            <div className="absolute right-3 bottom-3 flex items-center space-x-2">
              <Button
                variant="ghost"
                size="sm"
                className="p-1 h-auto text-gray-400 hover:text-gray-600"
              >
                <Paperclip className="w-4 h-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="p-1 h-auto text-gray-400 hover:text-gray-600"
              >
                <Mic className="w-4 h-4" />
              </Button>
            </div>
          </div>
          <Button
            onClick={handleSend}
            disabled={!message.trim() || isLoading}
            className="px-6 py-3 bg-primary hover:bg-primary/90 text-white rounded-2xl"
          >
            <Send className="w-4 h-4 mr-2" />
            Send
          </Button>
        </div>

        {/* Helper Text */}
        <div className="mt-2 text-xs text-gray-500 flex items-center justify-between">
          <span>Press Enter to send, Shift+Enter for new line</span>
          <span className="flex items-center space-x-1">
            <span className="w-2 h-2 bg-green-500 rounded-full"></span>
            <span>Powered by GPT-4 • Secure & Private</span>
          </span>
        </div>
      </div>
    </div>
  );
}
