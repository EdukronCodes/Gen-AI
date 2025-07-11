import { useState, useRef, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Bot, User as UserIcon, Settings, Trash2, Download, ShoppingCart, Truck, RotateCcw, Star, FileText, Search, MessageCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { User, Chat } from "@shared/schema";
import MessageBubble from "./message-bubble";
import TypingIndicator from "./typing-indicator";
import ChatInput from "./chat-input";
import ProductRecommendations from "./product-recommendations";
import OrderTrackingCard from "./order-tracking-card";

interface ChatWindowProps {
  user: User;
  chatHistory: Chat[];
}

interface Message {
  id: string;
  content: string;
  type: "user" | "bot";
  timestamp: Date;
  components?: {
    type: "order_tracking" | "product_recommendations";
    data: any;
  }[];
}

export default function ChatWindow({ user, chatHistory }: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [activeCategory, setActiveCategory] = useState("chat");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  useEffect(() => {
    // Initialize with welcome message
    const welcomeMessage: Message = {
      id: "welcome",
      content: `👋 Hi ${user.name}! I'm your AI customer support assistant. I'm here to help you with product information, order tracking, returns, recommendations, and store policies. How can I assist you today?`,
      type: "bot",
      timestamp: new Date(),
    };
    setMessages([welcomeMessage]);
  }, [user.name]);

  const sendMessageMutation = useMutation({
    mutationFn: async (message: string) => {
      const response = await apiRequest("POST", "/api/chat", {
        message,
        userId: user.id,
      });
      return response.json();
    },
    onMutate: (message) => {
      const userMessage: Message = {
        id: Date.now().toString(),
        content: message,
        type: "user",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, userMessage]);
      setIsTyping(true);
    },
    onSuccess: (data) => {
      setIsTyping(false);
      
      const botMessage: Message = {
        id: Date.now().toString() + "_bot",
        content: data.message,
        type: "bot",
        timestamp: new Date(),
      };

      // Add special components based on intent
      if (data.intent === "order_tracking") {
        botMessage.components = [
          {
            type: "order_tracking",
            data: { orderId: user.lastOrderId },
          },
        ];
      } else if (data.intent === "recommendations") {
        botMessage.components = [
          {
            type: "product_recommendations",
            data: { userId: user.id },
          },
        ];
      }

      setMessages(prev => [...prev, botMessage]);
      
      // Invalidate chat history to refetch
      queryClient.invalidateQueries({ queryKey: ["/api/chat/history", user.id] });
    },
    onError: (error) => {
      setIsTyping(false);
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleSendMessage = (message: string) => {
    if (message.trim()) {
      sendMessageMutation.mutate(message);
    }
  };

  const handleQuickAction = (action: string) => {
    const actionMessages = {
      "track-order": "I need to track my order. Can you help me with that?",
      "return-item": "I would like to return an item from my recent purchase.",
      "product-search": "Can you help me find specific products?",
      "recommendations": "What products would you recommend for me?",
    };

    const message = actionMessages[action as keyof typeof actionMessages];
    if (message) {
      handleSendMessage(message);
    }
  };

  const handleClearChat = () => {
    const welcomeMessage: Message = {
      id: "welcome",
      content: `👋 Hi ${user.name}! I'm your AI customer support assistant. I'm here to help you with product information, order tracking, returns, recommendations, and store policies. How can I assist you today?`,
      type: "bot",
      timestamp: new Date(),
    };
    setMessages([welcomeMessage]);
  };

  const categories = [
    { id: "chat", label: "Live Chat", icon: MessageCircle, active: true },
    { id: "products", label: "Product Info", icon: ShoppingCart, active: false },
    { id: "orders", label: "Order Tracking", icon: Truck, active: false },
    { id: "returns", label: "Returns & Refunds", icon: RotateCcw, active: false },
    { id: "recommendations", label: "Recommendations", icon: Star, active: false },
    { id: "policies", label: "Store Policies", icon: FileText, active: false },
  ];

  return (
    <>
      {/* Sidebar */}
      <aside className="w-80 sidebar-bg text-white flex flex-col">
        {/* Brand Header */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
              <Bot className="text-white text-xl" />
            </div>
            <div>
              <h1 className="text-xl font-bold">RetailBot</h1>
              <p className="text-gray-400 text-sm">AI Customer Support</p>
            </div>
          </div>
        </div>

        {/* User Profile */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center">
              <span className="text-white font-semibold text-lg">
                {user.name.split(' ').map(n => n[0]).join('')}
              </span>
            </div>
            <div>
              <h3 className="font-semibold">{user.name}</h3>
              <p className="text-gray-400 text-sm">Premium Customer</p>
            </div>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Last Order:</span>
              <span>{user.lastOrderId || "None"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Location:</span>
              <span>{user.location || "Unknown"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Category:</span>
              <span>{user.preferredCategory || "General"}</span>
            </div>
          </div>
        </div>

        {/* Support Categories */}
        <nav className="flex-1 p-6">
          <h4 className="text-gray-400 text-sm font-medium mb-4 uppercase tracking-wide">
            Support Categories
          </h4>
          <ul className="space-y-2">
            {categories.map((category) => {
              const Icon = category.icon;
              return (
                <li key={category.id}>
                  <button
                    onClick={() => setActiveCategory(category.id)}
                    className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-colors ${
                      activeCategory === category.id
                        ? "bg-primary bg-opacity-20 text-blue-300"
                        : "hover:bg-gray-700"
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span>{category.label}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Quick Actions */}
        <div className="p-6 border-t border-gray-700">
          <h4 className="text-gray-400 text-sm font-medium mb-3">Quick Actions</h4>
          <div className="space-y-2">
            <button
              onClick={() => handleQuickAction("track-order")}
              className="w-full text-left p-2 rounded-lg hover:bg-gray-700 transition-colors text-sm"
            >
              <Search className="inline w-4 h-4 mr-2" />
              Track My Order
            </button>
            <button
              onClick={() => handleQuickAction("return-item")}
              className="w-full text-left p-2 rounded-lg hover:bg-gray-700 transition-colors text-sm"
            >
              <RotateCcw className="inline w-4 h-4 mr-2" />
              Return an Item
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col">
        {/* Chat Header */}
        <header className="bg-white border-b border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center">
                <Bot className="text-white" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900">AI Customer Support</h2>
                <div className="flex items-center space-x-2 text-sm text-gray-500">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span>Online • Powered by GPT-4</span>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleClearChat}
                className="text-gray-400 hover:text-gray-600"
              >
                <Trash2 className="w-4 h-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-gray-400 hover:text-gray-600"
              >
                <Download className="w-4 h-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-gray-400 hover:text-gray-600"
              >
                <Settings className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col chat-bg">
          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map((message) => (
              <div key={message.id}>
                <MessageBubble
                  message={message.content}
                  type={message.type}
                  timestamp={message.timestamp}
                  userInitials={user.name.split(' ').map(n => n[0]).join('')}
                />
                {message.components?.map((component, index) => (
                  <div key={index} className="mt-4">
                    {component.type === "order_tracking" && (
                      <OrderTrackingCard orderId={component.data.orderId} />
                    )}
                    {component.type === "product_recommendations" && (
                      <ProductRecommendations userId={component.data.userId} />
                    )}
                  </div>
                ))}
              </div>
            ))}
            
            {isTyping && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>

          {/* Chat Input */}
          <ChatInput
            onSendMessage={handleSendMessage}
            onQuickAction={handleQuickAction}
            isLoading={sendMessageMutation.isPending}
          />
        </div>
      </main>
    </>
  );
}
