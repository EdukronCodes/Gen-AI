import OpenAI from "openai";

// the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY_ENV_VAR || "your-api-key-here",
});

export interface ChatContext {
  userName: string;
  lastOrder?: string;
  location?: string;
  preferredCategory?: string;
  chatHistory?: Array<{ message: string; response: string; timestamp: Date }>;
}

export interface ChatResponse {
  message: string;
  intent: string;
  confidence: number;
  suggestedActions?: string[];
}

export async function generateChatResponse(
  userMessage: string,
  context: ChatContext
): Promise<ChatResponse> {
  // Check if OpenAI API key is available and valid
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey || apiKey === "your-api-key-here") {
    return generateFallbackResponse(userMessage, context);
  }

  try {
    const systemPrompt = `You are RetailBot, an AI-powered customer support assistant for an online retail store. You help customers with:

1. Product Information (availability, features, prices, specifications)
2. Order Tracking (based on order ID or email)
3. Returns and Refund Policies
4. Personalized Product Recommendations (based on past orders or interests)
5. Store Policies (shipping, delivery, cancellation)

Always respond in a friendly, concise, and helpful tone. Use the customer's context to provide personalized assistance.

Current customer context:
- Name: ${context.userName}
- Last order: ${context.lastOrder || "None"}
- Location: ${context.location || "Unknown"}
- Preferred category: ${context.preferredCategory || "General"}

Recent chat history:
${context.chatHistory?.slice(-3).map(chat => `User: ${chat.message}\nBot: ${chat.response}`).join('\n') || "No recent history"}

Respond with JSON in this format:
{
  "message": "Your helpful response to the customer",
  "intent": "product_info|order_tracking|returns|recommendations|policies|general_support",
  "confidence": 0.95,
  "suggestedActions": ["action1", "action2"]
}`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage }
      ],
      response_format: { type: "json_object" },
      temperature: 0.7,
      max_tokens: 500,
    });

    const result = JSON.parse(response.choices[0].message.content || "{}");
    
    return {
      message: result.message || "I'm sorry, I couldn't process your request. Please try again.",
      intent: result.intent || "general_support",
      confidence: Math.max(0, Math.min(1, result.confidence || 0.5)),
      suggestedActions: result.suggestedActions || [],
    };
  } catch (error) {
    console.error("OpenAI API Error:", error);
    return generateFallbackResponse(userMessage, context);
  }
}

// Fallback response system when OpenAI is not available
function generateFallbackResponse(userMessage: string, context: ChatContext): ChatResponse {
  const message = userMessage.toLowerCase();
  
  // Order tracking patterns
  if (message.includes("order") || message.includes("track") || message.includes("shipping")) {
    return {
      message: `Hi ${context.userName}! I can help you track your order. ${context.lastOrder ? `Your last order ${context.lastOrder} is currently being processed.` : "Please provide your order ID to get the latest status."} You can also check the order tracking section for detailed updates.`,
      intent: "order_tracking",
      confidence: 0.8,
      suggestedActions: ["track_order", "view_order_history"],
    };
  }
  
  // Return/refund patterns
  if (message.includes("return") || message.includes("refund") || message.includes("cancel")) {
    return {
      message: `I understand you'd like to return an item. Our return policy allows returns within 30 days of purchase. Items must be in original condition with tags attached. Would you like me to start the return process or provide more details about our return policy?`,
      intent: "returns",
      confidence: 0.85,
      suggestedActions: ["start_return", "view_return_policy"],
    };
  }
  
  // Product search patterns
  if (message.includes("product") || message.includes("search") || message.includes("find") || message.includes("looking for")) {
    return {
      message: `I'd be happy to help you find products! ${context.preferredCategory ? `Based on your interest in ${context.preferredCategory}, I can show you our latest items in that category.` : "Let me know what you're looking for and I'll help you find the perfect product."} You can also browse by category or use our search feature.`,
      intent: "product_info",
      confidence: 0.75,
      suggestedActions: ["search_products", "browse_categories"],
    };
  }
  
  // Recommendation patterns
  if (message.includes("recommend") || message.includes("suggest") || message.includes("what should")) {
    return {
      message: `Based on your previous purchases and interests in ${context.preferredCategory || "various categories"}, I can suggest some great products for you! Let me pull up some personalized recommendations that match your style and preferences.`,
      intent: "recommendations",
      confidence: 0.8,
      suggestedActions: ["view_recommendations", "browse_categories"],
    };
  }
  
  // Policy/help patterns
  if (message.includes("policy") || message.includes("help") || message.includes("support")) {
    return {
      message: `I'm here to help you with any questions about our store policies, shipping, returns, or general support. What specific information would you like to know about? I can provide details about our shipping policies, return procedures, or any other store-related questions.`,
      intent: "policies",
      confidence: 0.7,
      suggestedActions: ["view_policies", "contact_support"],
    };
  }
  
  // Greeting patterns
  if (message.includes("hello") || message.includes("hi") || message.includes("hey")) {
    return {
      message: `Hello ${context.userName}! Welcome to our customer support. I'm here to help you with product information, order tracking, returns, recommendations, and any questions about our store policies. What can I assist you with today?`,
      intent: "general_support",
      confidence: 0.9,
      suggestedActions: ["track_order", "search_products", "view_recommendations"],
    };
  }
  
  // Default response
  return {
    message: `Hi ${context.userName}! I'm here to help you with any questions about our products, orders, returns, or store policies. Could you please provide more details about what you're looking for? I can help you track orders, find products, process returns, or provide personalized recommendations.`,
    intent: "general_support",
    confidence: 0.6,
    suggestedActions: ["track_order", "search_products", "view_recommendations", "contact_support"],
  };
}

export async function generateProductRecommendations(
  userContext: ChatContext,
  preferredCategory?: string
): Promise<{
  recommendations: Array<{
    productId: number;
    reason: string;
    confidence: number;
  }>;
}> {
  // Check if OpenAI API key is available and valid
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey || apiKey === "your-api-key-here") {
    return generateFallbackRecommendations(userContext, preferredCategory);
  }

  try {
    const prompt = `Based on the customer profile below, suggest 3 product recommendations with reasons and confidence scores.

Customer Profile:
- Name: ${userContext.userName}
- Last order: ${userContext.lastOrder || "None"}
- Location: ${userContext.location || "Unknown"}
- Preferred category: ${preferredCategory || userContext.preferredCategory || "General"}

Available categories: Electronics, Home & Garden, Fashion, Sports, Books

Respond with JSON in this format:
{
  "recommendations": [
    {
      "productId": 1,
      "reason": "Based on your interest in electronics and previous purchases",
      "confidence": 0.85
    }
  ]
}`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: "You are a product recommendation expert for an online retail store." },
        { role: "user", content: prompt }
      ],
      response_format: { type: "json_object" },
      temperature: 0.6,
      max_tokens: 300,
    });

    const result = JSON.parse(response.choices[0].message.content || "{}");
    
    return {
      recommendations: result.recommendations || [],
    };
  } catch (error) {
    console.error("OpenAI API Error:", error);
    return generateFallbackRecommendations(userContext, preferredCategory);
  }
}

// Fallback recommendation system when OpenAI is not available
function generateFallbackRecommendations(
  userContext: ChatContext,
  preferredCategory?: string
): {
  recommendations: Array<{
    productId: number;
    reason: string;
    confidence: number;
  }>;
} {
  const category = preferredCategory || userContext.preferredCategory || "Electronics";
  
  // Basic recommendation logic based on category preference
  const recommendations = [];
  
  if (category.toLowerCase() === "electronics") {
    recommendations.push(
      {
        productId: 1,
        reason: "Popular noise-cancelling headphones perfect for daily use",
        confidence: 0.85
      },
      {
        productId: 2,
        reason: "Essential charging station for multiple devices",
        confidence: 0.80
      },
      {
        productId: 3,
        reason: "Smart fitness watch for health tracking",
        confidence: 0.75
      }
    );
  } else if (category.toLowerCase() === "fashion") {
    recommendations.push(
      {
        productId: 5,
        reason: "Comfortable organic cotton t-shirt, great for casual wear",
        confidence: 0.80
      },
      {
        productId: 1,
        reason: "Also recommended: premium headphones for music lovers",
        confidence: 0.70
      },
      {
        productId: 6,
        reason: "Versatile yoga mat for fitness and relaxation",
        confidence: 0.65
      }
    );
  } else if (category.toLowerCase() === "sports") {
    recommendations.push(
      {
        productId: 6,
        reason: "High-quality yoga mat for all fitness levels",
        confidence: 0.85
      },
      {
        productId: 3,
        reason: "Fitness watch to track your workouts and health",
        confidence: 0.80
      },
      {
        productId: 5,
        reason: "Comfortable workout t-shirt made from organic cotton",
        confidence: 0.75
      }
    );
  } else {
    // Default mixed recommendations
    recommendations.push(
      {
        productId: 1,
        reason: "Top-rated wireless headphones with excellent battery life",
        confidence: 0.80
      },
      {
        productId: 5,
        reason: "Essential organic cotton t-shirt for everyday comfort",
        confidence: 0.75
      },
      {
        productId: 2,
        reason: "Convenient charging station for all your devices",
        confidence: 0.70
      }
    );
  }
  
  return { recommendations };
}
