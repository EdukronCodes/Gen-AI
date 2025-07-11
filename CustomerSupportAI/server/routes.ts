import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { generateChatResponse, generateProductRecommendations } from "./services/openai";
import { insertChatSchema } from "@shared/schema";
import { z } from "zod";

export async function registerRoutes(app: Express): Promise<Server> {
  // Chat endpoint
  app.post("/api/chat", async (req, res) => {
    try {
      const { message, userId } = req.body;
      
      if (!message || !userId) {
        return res.status(400).json({ error: "Message and userId are required" });
      }

      // Get user context
      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      // Get recent chat history for context
      const recentChats = await storage.getRecentChats(userId, 5);
      
      // Generate AI response
      const aiResponse = await generateChatResponse(message, {
        userName: user.name,
        lastOrder: user.lastOrderId || undefined,
        location: user.location || undefined,
        preferredCategory: user.preferredCategory || undefined,
        chatHistory: recentChats.map(chat => ({
          message: chat.message,
          response: chat.response,
          timestamp: chat.timestamp,
        })),
      });

      // Save chat to storage
      await storage.createChat({
        userId,
        message,
        response: aiResponse.message,
        messageType: "conversation",
        context: JSON.stringify({
          intent: aiResponse.intent,
          confidence: aiResponse.confidence,
          suggestedActions: aiResponse.suggestedActions,
        }),
      });

      res.json(aiResponse);
    } catch (error) {
      console.error("Chat error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Get chat history
  app.get("/api/chat/history/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const chats = await storage.getChatHistory(userId);
      res.json(chats);
    } catch (error) {
      console.error("Chat history error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Product search
  app.get("/api/products/search", async (req, res) => {
    try {
      const { q, category } = req.query;
      
      let products;
      if (q) {
        products = await storage.searchProducts(q as string);
      } else if (category) {
        products = await storage.getProductsByCategory(category as string);
      } else {
        products = await storage.getProducts();
      }

      res.json(products);
    } catch (error) {
      console.error("Product search error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Get product by ID
  app.get("/api/products/:id", async (req, res) => {
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

  // Order tracking
  app.get("/api/orders/:orderId", async (req, res) => {
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

  // Get orders by user
  app.get("/api/orders/user/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const orders = await storage.getOrdersByUser(userId);
      res.json(orders);
    } catch (error) {
      console.error("User orders error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Get recommendations
  app.get("/api/recommendations/:userId", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      const { category } = req.query;
      
      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      const recommendations = await generateProductRecommendations({
        userName: user.name,
        lastOrder: user.lastOrderId || undefined,
        location: user.location || undefined,
        preferredCategory: user.preferredCategory || undefined,
      }, category as string);

      // Get actual product details for recommendations
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

  // Get user profile
  app.get("/api/users/:id", async (req, res) => {
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

  // Search products by query
  app.get("/api/products/search/:query", async (req, res) => {
    try {
      const query = req.params.query;
      const products = await storage.searchProducts(query);
      res.json(products);
    } catch (error) {
      console.error("Product search error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Get products by category
  app.get("/api/products/category/:category", async (req, res) => {
    try {
      const category = req.params.category;
      const products = await storage.getProductsByCategory(category);
      res.json(products);
    } catch (error) {
      console.error("Category products error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Get all products
  app.get("/api/products", async (req, res) => {
    try {
      const products = await storage.getProducts();
      res.json(products);
    } catch (error) {
      console.error("Get all products error:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  // Health check endpoint
  app.get("/api/health", (req, res) => {
    res.json({ 
      status: "healthy", 
      timestamp: new Date().toISOString(),
      services: {
        database: "connected",
        ai: process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY !== "your-api-key-here" ? "connected" : "fallback_mode"
      }
    });
  });

  const httpServer = createServer(app);
  return httpServer;
}
