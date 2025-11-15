// server/index.ts
import express2 from "express";

// server/routes.ts
import { createServer } from "http";

// server/storage.ts
var MemStorage = class {
  users;
  products;
  orders;
  chats;
  currentUserId;
  currentProductId;
  currentOrderId;
  currentChatId;
  constructor() {
    this.users = /* @__PURE__ */ new Map();
    this.products = /* @__PURE__ */ new Map();
    this.orders = /* @__PURE__ */ new Map();
    this.chats = /* @__PURE__ */ new Map();
    this.currentUserId = 1;
    this.currentProductId = 1;
    this.currentOrderId = 1;
    this.currentChatId = 1;
    this.initializeMockData();
  }
  initializeMockData() {
    const defaultUser = {
      id: 1,
      username: "johndoe",
      email: "john@example.com",
      name: "John Doe",
      location: "New York, NY",
      lastOrderId: "ORD-2024-001",
      preferredCategory: "Electronics",
      createdAt: /* @__PURE__ */ new Date()
    };
    this.users.set(1, defaultUser);
    this.currentUserId = 2;
    const sampleProducts = [
      {
        id: 1,
        name: "Wireless Bluetooth Headphones",
        description: "Premium noise-cancelling audio headphones with 30-hour battery life",
        price: "149.99",
        category: "Electronics",
        imageUrl: "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=400&h=300",
        inStock: true,
        specifications: ["Bluetooth 5.0", "30-hour battery", "Active noise cancellation"],
        createdAt: /* @__PURE__ */ new Date()
      },
      {
        id: 2,
        name: "Multi-Port Charging Station",
        description: "Fast charging station with 6 USB ports and wireless charging pad",
        price: "79.99",
        category: "Electronics",
        imageUrl: "https://images.unsplash.com/photo-1583394838336-acd977736f90?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300",
        inStock: true,
        specifications: ["6 USB ports", "Wireless charging", "Quick charge 3.0"],
        createdAt: /* @__PURE__ */ new Date()
      },
      {
        id: 3,
        name: "Smart Fitness Watch",
        description: "Advanced health monitoring with GPS and heart rate tracking",
        price: "299.99",
        category: "Electronics",
        imageUrl: "https://images.unsplash.com/photo-1523275335684-37898b6baf30?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=400&h=300",
        inStock: true,
        specifications: ["GPS tracking", "Heart rate monitor", "7-day battery"],
        createdAt: /* @__PURE__ */ new Date()
      },
      {
        id: 4,
        name: "Gaming Mechanical Keyboard",
        description: "RGB backlit mechanical keyboard with tactile switches",
        price: "129.99",
        category: "Electronics",
        imageUrl: "https://images.unsplash.com/photo-1541140532154-b024d705b90a?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300",
        inStock: true,
        specifications: ["RGB backlit", "Mechanical switches", "Anti-ghosting"],
        createdAt: /* @__PURE__ */ new Date()
      },
      {
        id: 5,
        name: "Cotton Blend T-Shirt",
        description: "Comfortable casual t-shirt made from organic cotton blend",
        price: "29.99",
        category: "Fashion",
        imageUrl: "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300",
        inStock: true,
        specifications: ["100% organic cotton", "Pre-shrunk", "Available in multiple colors"],
        createdAt: /* @__PURE__ */ new Date()
      },
      {
        id: 6,
        name: "Yoga Mat Premium",
        description: "High-quality non-slip yoga mat for all fitness levels",
        price: "39.99",
        category: "Sports",
        imageUrl: "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300",
        inStock: true,
        specifications: ["Non-slip surface", "6mm thickness", "Eco-friendly material"],
        createdAt: /* @__PURE__ */ new Date()
      }
    ];
    sampleProducts.forEach((product) => {
      this.products.set(product.id, product);
    });
    this.currentProductId = 7;
    const sampleOrder = {
      id: 1,
      orderId: "ORD-2024-001",
      userId: 1,
      status: "shipped",
      trackingNumber: "1Z999AA123456789",
      orderDate: /* @__PURE__ */ new Date("2024-12-15"),
      estimatedDelivery: /* @__PURE__ */ new Date("2024-12-18"),
      totalAmount: "149.99",
      items: ['{"productId": 1, "quantity": 1, "price": "149.99"}']
    };
    this.orders.set("ORD-2024-001", sampleOrder);
  }
  // User operations
  async getUser(id) {
    return this.users.get(id);
  }
  async getUserByEmail(email) {
    return Array.from(this.users.values()).find((user) => user.email === email);
  }
  async createUser(insertUser) {
    const id = this.currentUserId++;
    const user = {
      ...insertUser,
      id,
      location: insertUser.location || null,
      lastOrderId: insertUser.lastOrderId || null,
      preferredCategory: insertUser.preferredCategory || null,
      createdAt: /* @__PURE__ */ new Date()
    };
    this.users.set(id, user);
    return user;
  }
  async updateUser(id, updates) {
    const user = this.users.get(id);
    if (!user) return void 0;
    const updatedUser = { ...user, ...updates };
    this.users.set(id, updatedUser);
    return updatedUser;
  }
  // Product operations
  async getProducts() {
    return Array.from(this.products.values());
  }
  async getProduct(id) {
    return this.products.get(id);
  }
  async searchProducts(query) {
    const searchTerm = query.toLowerCase();
    return Array.from(this.products.values()).filter(
      (product) => product.name.toLowerCase().includes(searchTerm) || product.description.toLowerCase().includes(searchTerm) || product.category.toLowerCase().includes(searchTerm)
    );
  }
  async getProductsByCategory(category) {
    return Array.from(this.products.values()).filter(
      (product) => product.category.toLowerCase() === category.toLowerCase()
    );
  }
  async createProduct(insertProduct) {
    const id = this.currentProductId++;
    const product = {
      ...insertProduct,
      id,
      imageUrl: insertProduct.imageUrl || null,
      inStock: insertProduct.inStock ?? true,
      specifications: insertProduct.specifications || null,
      createdAt: /* @__PURE__ */ new Date()
    };
    this.products.set(id, product);
    return product;
  }
  // Order operations
  async getOrder(orderId) {
    return this.orders.get(orderId);
  }
  async getOrdersByUser(userId) {
    return Array.from(this.orders.values()).filter((order) => order.userId === userId);
  }
  async createOrder(insertOrder) {
    const id = this.currentOrderId++;
    const order = {
      ...insertOrder,
      id,
      trackingNumber: insertOrder.trackingNumber || null,
      estimatedDelivery: insertOrder.estimatedDelivery || null,
      orderDate: /* @__PURE__ */ new Date()
    };
    this.orders.set(insertOrder.orderId, order);
    return order;
  }
  async updateOrderStatus(orderId, status) {
    const order = this.orders.get(orderId);
    if (!order) return void 0;
    const updatedOrder = { ...order, status };
    this.orders.set(orderId, updatedOrder);
    return updatedOrder;
  }
  // Chat operations
  async getChatHistory(userId) {
    return Array.from(this.chats.values()).filter((chat) => chat.userId === userId).sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }
  async createChat(insertChat) {
    const id = this.currentChatId++;
    const chat = {
      ...insertChat,
      id,
      context: insertChat.context || null,
      timestamp: /* @__PURE__ */ new Date()
    };
    this.chats.set(id, chat);
    return chat;
  }
  async getRecentChats(userId, limit) {
    return Array.from(this.chats.values()).filter((chat) => chat.userId === userId).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()).slice(0, limit);
  }
};
var storage = new MemStorage();

// server/services/openai.ts
import OpenAI from "openai";
var openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY_ENV_VAR || "your-api-key-here"
});
async function generateChatResponse(userMessage, context) {
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
${context.chatHistory?.slice(-3).map((chat) => `User: ${chat.message}
Bot: ${chat.response}`).join("\n") || "No recent history"}

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
      max_tokens: 500
    });
    const result = JSON.parse(response.choices[0].message.content || "{}");
    return {
      message: result.message || "I'm sorry, I couldn't process your request. Please try again.",
      intent: result.intent || "general_support",
      confidence: Math.max(0, Math.min(1, result.confidence || 0.5)),
      suggestedActions: result.suggestedActions || []
    };
  } catch (error) {
    console.error("OpenAI API Error:", error);
    return generateFallbackResponse(userMessage, context);
  }
}
function generateFallbackResponse(userMessage, context) {
  const message = userMessage.toLowerCase();
  if (message.includes("order") || message.includes("track") || message.includes("shipping")) {
    return {
      message: `Hi ${context.userName}! I can help you track your order. ${context.lastOrder ? `Your last order ${context.lastOrder} is currently being processed.` : "Please provide your order ID to get the latest status."} You can also check the order tracking section for detailed updates.`,
      intent: "order_tracking",
      confidence: 0.8,
      suggestedActions: ["track_order", "view_order_history"]
    };
  }
  if (message.includes("return") || message.includes("refund") || message.includes("cancel")) {
    return {
      message: `I understand you'd like to return an item. Our return policy allows returns within 30 days of purchase. Items must be in original condition with tags attached. Would you like me to start the return process or provide more details about our return policy?`,
      intent: "returns",
      confidence: 0.85,
      suggestedActions: ["start_return", "view_return_policy"]
    };
  }
  if (message.includes("product") || message.includes("search") || message.includes("find") || message.includes("looking for")) {
    return {
      message: `I'd be happy to help you find products! ${context.preferredCategory ? `Based on your interest in ${context.preferredCategory}, I can show you our latest items in that category.` : "Let me know what you're looking for and I'll help you find the perfect product."} You can also browse by category or use our search feature.`,
      intent: "product_info",
      confidence: 0.75,
      suggestedActions: ["search_products", "browse_categories"]
    };
  }
  if (message.includes("recommend") || message.includes("suggest") || message.includes("what should")) {
    return {
      message: `Based on your previous purchases and interests in ${context.preferredCategory || "various categories"}, I can suggest some great products for you! Let me pull up some personalized recommendations that match your style and preferences.`,
      intent: "recommendations",
      confidence: 0.8,
      suggestedActions: ["view_recommendations", "browse_categories"]
    };
  }
  if (message.includes("policy") || message.includes("help") || message.includes("support")) {
    return {
      message: `I'm here to help you with any questions about our store policies, shipping, returns, or general support. What specific information would you like to know about? I can provide details about our shipping policies, return procedures, or any other store-related questions.`,
      intent: "policies",
      confidence: 0.7,
      suggestedActions: ["view_policies", "contact_support"]
    };
  }
  if (message.includes("hello") || message.includes("hi") || message.includes("hey")) {
    return {
      message: `Hello ${context.userName}! Welcome to our customer support. I'm here to help you with product information, order tracking, returns, recommendations, and any questions about our store policies. What can I assist you with today?`,
      intent: "general_support",
      confidence: 0.9,
      suggestedActions: ["track_order", "search_products", "view_recommendations"]
    };
  }
  return {
    message: `Hi ${context.userName}! I'm here to help you with any questions about our products, orders, returns, or store policies. Could you please provide more details about what you're looking for? I can help you track orders, find products, process returns, or provide personalized recommendations.`,
    intent: "general_support",
    confidence: 0.6,
    suggestedActions: ["track_order", "search_products", "view_recommendations", "contact_support"]
  };
}
async function generateProductRecommendations(userContext, preferredCategory) {
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
      max_tokens: 300
    });
    const result = JSON.parse(response.choices[0].message.content || "{}");
    return {
      recommendations: result.recommendations || []
    };
  } catch (error) {
    console.error("OpenAI API Error:", error);
    return generateFallbackRecommendations(userContext, preferredCategory);
  }
}
function generateFallbackRecommendations(userContext, preferredCategory) {
  const category = preferredCategory || userContext.preferredCategory || "Electronics";
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
        confidence: 0.8
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
        confidence: 0.8
      },
      {
        productId: 1,
        reason: "Also recommended: premium headphones for music lovers",
        confidence: 0.7
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
        confidence: 0.8
      },
      {
        productId: 5,
        reason: "Comfortable workout t-shirt made from organic cotton",
        confidence: 0.75
      }
    );
  } else {
    recommendations.push(
      {
        productId: 1,
        reason: "Top-rated wireless headphones with excellent battery life",
        confidence: 0.8
      },
      {
        productId: 5,
        reason: "Essential organic cotton t-shirt for everyday comfort",
        confidence: 0.75
      },
      {
        productId: 2,
        reason: "Convenient charging station for all your devices",
        confidence: 0.7
      }
    );
  }
  return { recommendations };
}

// server/routes.ts
async function registerRoutes(app2) {
  app2.post("/api/chat", async (req, res) => {
    try {
      const { message, userId } = req.body;
      if (!message || !userId) {
        return res.status(400).json({ error: "Message and userId are required" });
      }
      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }
      const recentChats = await storage.getRecentChats(userId, 5);
      const aiResponse = await generateChatResponse(message, {
        userName: user.name,
        lastOrder: user.lastOrderId || void 0,
        location: user.location || void 0,
        preferredCategory: user.preferredCategory || void 0,
        chatHistory: recentChats.map((chat) => ({
          message: chat.message,
          response: chat.response,
          timestamp: chat.timestamp
        }))
      });
      await storage.createChat({
        userId,
        message,
        response: aiResponse.message,
        messageType: "conversation",
        context: JSON.stringify({
          intent: aiResponse.intent,
          confidence: aiResponse.confidence,
          suggestedActions: aiResponse.suggestedActions
        })
      });
      res.json(aiResponse);
    } catch (error) {
      console.error("Chat error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/chat/history/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const chats = await storage.getChatHistory(userId);
      res.json(chats);
    } catch (error) {
      console.error("Chat history error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/products/search", async (req, res) => {
    try {
      const { q, category } = req.query;
      let products;
      if (q) {
        products = await storage.searchProducts(q);
      } else if (category) {
        products = await storage.getProductsByCategory(category);
      } else {
        products = await storage.getProducts();
      }
      res.json(products);
    } catch (error) {
      console.error("Product search error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/products/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const product = await storage.getProduct(id);
      if (!product) {
        return res.status(404).json({ error: "Product not found" });
      }
      res.json(product);
    } catch (error) {
      console.error("Product fetch error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/orders/:orderId", async (req, res) => {
    try {
      const orderId = req.params.orderId;
      const order = await storage.getOrder(orderId);
      if (!order) {
        return res.status(404).json({ error: "Order not found" });
      }
      res.json(order);
    } catch (error) {
      console.error("Order tracking error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/orders/user/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const orders = await storage.getOrdersByUser(userId);
      res.json(orders);
    } catch (error) {
      console.error("User orders error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/recommendations/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const { category } = req.query;
      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }
      const recommendations = await generateProductRecommendations({
        userName: user.name,
        lastOrder: user.lastOrderId || void 0,
        location: user.location || void 0,
        preferredCategory: user.preferredCategory || void 0
      }, category);
      const productPromises = recommendations.recommendations.map(async (rec) => {
        const product = await storage.getProduct(rec.productId);
        return product ? { ...product, recommendationReason: rec.reason, confidence: rec.confidence } : null;
      });
      const products = (await Promise.all(productPromises)).filter(Boolean);
      res.json(products);
    } catch (error) {
      console.error("Recommendations error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/users/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const user = await storage.getUser(id);
      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }
      res.json(user);
    } catch (error) {
      console.error("User fetch error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/products/search/:query", async (req, res) => {
    try {
      const query = req.params.query;
      const products = await storage.searchProducts(query);
      res.json(products);
    } catch (error) {
      console.error("Product search error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/products/category/:category", async (req, res) => {
    try {
      const category = req.params.category;
      const products = await storage.getProductsByCategory(category);
      res.json(products);
    } catch (error) {
      console.error("Category products error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/products", async (req, res) => {
    try {
      const products = await storage.getProducts();
      res.json(products);
    } catch (error) {
      console.error("Get all products error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  app2.get("/api/health", (req, res) => {
    res.json({
      status: "healthy",
      timestamp: (/* @__PURE__ */ new Date()).toISOString(),
      services: {
        database: "connected",
        ai: process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY !== "your-api-key-here" ? "connected" : "fallback_mode"
      }
    });
  });
  const httpServer = createServer(app2);
  return httpServer;
}

// server/vite.ts
import express from "express";
import fs from "fs";
import path2 from "path";
import { createServer as createViteServer, createLogger } from "vite";

// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import runtimeErrorOverlay from "@replit/vite-plugin-runtime-error-modal";
var vite_config_default = defineConfig({
  plugins: [
    react(),
    runtimeErrorOverlay(),
    ...process.env.NODE_ENV !== "production" && process.env.REPL_ID !== void 0 ? [
      await import("@replit/vite-plugin-cartographer").then(
        (m) => m.cartographer()
      )
    ] : []
  ],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "client", "src"),
      "@shared": path.resolve(import.meta.dirname, "shared"),
      "@assets": path.resolve(import.meta.dirname, "attached_assets")
    }
  },
  root: path.resolve(import.meta.dirname, "client"),
  build: {
    outDir: path.resolve(import.meta.dirname, "dist/public"),
    emptyOutDir: true
  },
  server: {
    fs: {
      strict: true,
      deny: ["**/.*"]
    }
  }
});

// server/vite.ts
import { nanoid } from "nanoid";
var viteLogger = createLogger();
function log(message, source = "express") {
  const formattedTime = (/* @__PURE__ */ new Date()).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true
  });
  console.log(`${formattedTime} [${source}] ${message}`);
}
async function setupVite(app2, server) {
  const serverOptions = {
    middlewareMode: true,
    hmr: { server },
    allowedHosts: true
  };
  const vite = await createViteServer({
    ...vite_config_default,
    configFile: false,
    customLogger: {
      ...viteLogger,
      error: (msg, options) => {
        viteLogger.error(msg, options);
        process.exit(1);
      }
    },
    server: serverOptions,
    appType: "custom"
  });
  app2.use(vite.middlewares);
  app2.use("*", async (req, res, next) => {
    const url = req.originalUrl;
    try {
      const clientTemplate = path2.resolve(
        import.meta.dirname,
        "..",
        "client",
        "index.html"
      );
      let template = await fs.promises.readFile(clientTemplate, "utf-8");
      template = template.replace(
        `src="/src/main.tsx"`,
        `src="/src/main.tsx?v=${nanoid()}"`
      );
      const page = await vite.transformIndexHtml(url, template);
      res.status(200).set({ "Content-Type": "text/html" }).end(page);
    } catch (e) {
      vite.ssrFixStacktrace(e);
      next(e);
    }
  });
}
function serveStatic(app2) {
  const distPath = path2.resolve(import.meta.dirname, "public");
  if (!fs.existsSync(distPath)) {
    throw new Error(
      `Could not find the build directory: ${distPath}, make sure to build the client first`
    );
  }
  app2.use(express.static(distPath));
  app2.use("*", (_req, res) => {
    res.sendFile(path2.resolve(distPath, "index.html"));
  });
}

// server/index.ts
var app = express2();
app.use(express2.json());
app.use(express2.urlencoded({ extended: false }));
app.use((req, res, next) => {
  const start = Date.now();
  const path3 = req.path;
  let capturedJsonResponse = void 0;
  const originalResJson = res.json;
  res.json = function(bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };
  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path3.startsWith("/api")) {
      let logLine = `${req.method} ${path3} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }
      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "\u2026";
      }
      log(logLine);
    }
  });
  next();
});
(async () => {
  const server = await registerRoutes(app);
  app.use((err, _req, res, _next) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";
    res.status(status).json({ message });
    throw err;
  });
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }
  const port = parseInt(process.env.PORT || "5000", 10);
  server.listen({
    port,
    host: "0.0.0.0",
    reusePort: true
  }, () => {
    log(`serving on port ${port}`);
  });
})();
