import { users, products, orders, chats, type User, type InsertUser, type Product, type InsertProduct, type Order, type InsertOrder, type Chat, type InsertChat } from "@shared/schema";

export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByEmail(email: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  updateUser(id: number, updates: Partial<InsertUser>): Promise<User | undefined>;

  // Product operations
  getProducts(): Promise<Product[]>;
  getProduct(id: number): Promise<Product | undefined>;
  searchProducts(query: string): Promise<Product[]>;
  getProductsByCategory(category: string): Promise<Product[]>;
  createProduct(product: InsertProduct): Promise<Product>;

  // Order operations
  getOrder(orderId: string): Promise<Order | undefined>;
  getOrdersByUser(userId: number): Promise<Order[]>;
  createOrder(order: InsertOrder): Promise<Order>;
  updateOrderStatus(orderId: string, status: string): Promise<Order | undefined>;

  // Chat operations
  getChatHistory(userId: number): Promise<Chat[]>;
  createChat(chat: InsertChat): Promise<Chat>;
  getRecentChats(userId: number, limit: number): Promise<Chat[]>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private products: Map<number, Product>;
  private orders: Map<string, Order>;
  private chats: Map<number, Chat>;
  private currentUserId: number;
  private currentProductId: number;
  private currentOrderId: number;
  private currentChatId: number;

  constructor() {
    this.users = new Map();
    this.products = new Map();
    this.orders = new Map();
    this.chats = new Map();
    this.currentUserId = 1;
    this.currentProductId = 1;
    this.currentOrderId = 1;
    this.currentChatId = 1;

    this.initializeMockData();
  }

  private initializeMockData() {
    // Create default user
    const defaultUser: User = {
      id: 1,
      username: "johndoe",
      email: "john@example.com",
      name: "John Doe",
      location: "New York, NY",
      lastOrderId: "ORD-2024-001",
      preferredCategory: "Electronics",
      createdAt: new Date(),
    };
    this.users.set(1, defaultUser);
    this.currentUserId = 2;

    // Create sample products
    const sampleProducts: Product[] = [
      {
        id: 1,
        name: "Wireless Bluetooth Headphones",
        description: "Premium noise-cancelling audio headphones with 30-hour battery life",
        price: "149.99",
        category: "Electronics",
        imageUrl: "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=400&h=300",
        inStock: true,
        specifications: ["Bluetooth 5.0", "30-hour battery", "Active noise cancellation"],
        createdAt: new Date(),
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
        createdAt: new Date(),
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
        createdAt: new Date(),
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
        createdAt: new Date(),
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
        createdAt: new Date(),
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
        createdAt: new Date(),
      },
    ];

    sampleProducts.forEach(product => {
      this.products.set(product.id, product);
    });
    this.currentProductId = 7;

    // Create sample order
    const sampleOrder: Order = {
      id: 1,
      orderId: "ORD-2024-001",
      userId: 1,
      status: "shipped",
      trackingNumber: "1Z999AA123456789",
      orderDate: new Date("2024-12-15"),
      estimatedDelivery: new Date("2024-12-18"),
      totalAmount: "149.99",
      items: ['{"productId": 1, "quantity": 1, "price": "149.99"}'],
    };
    this.orders.set("ORD-2024-001", sampleOrder);
  }

  // User operations
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByEmail(email: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(user => user.email === email);
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentUserId++;
    const user: User = {
      ...insertUser,
      id,
      location: insertUser.location || null,
      lastOrderId: insertUser.lastOrderId || null,
      preferredCategory: insertUser.preferredCategory || null,
      createdAt: new Date(),
    };
    this.users.set(id, user);
    return user;
  }

  async updateUser(id: number, updates: Partial<InsertUser>): Promise<User | undefined> {
    const user = this.users.get(id);
    if (!user) return undefined;

    const updatedUser = { ...user, ...updates };
    this.users.set(id, updatedUser);
    return updatedUser;
  }

  // Product operations
  async getProducts(): Promise<Product[]> {
    return Array.from(this.products.values());
  }

  async getProduct(id: number): Promise<Product | undefined> {
    return this.products.get(id);
  }

  async searchProducts(query: string): Promise<Product[]> {
    const searchTerm = query.toLowerCase();
    return Array.from(this.products.values()).filter(product =>
      product.name.toLowerCase().includes(searchTerm) ||
      product.description.toLowerCase().includes(searchTerm) ||
      product.category.toLowerCase().includes(searchTerm)
    );
  }

  async getProductsByCategory(category: string): Promise<Product[]> {
    return Array.from(this.products.values()).filter(product =>
      product.category.toLowerCase() === category.toLowerCase()
    );
  }

  async createProduct(insertProduct: InsertProduct): Promise<Product> {
    const id = this.currentProductId++;
    const product: Product = {
      ...insertProduct,
      id,
      imageUrl: insertProduct.imageUrl || null,
      inStock: insertProduct.inStock ?? true,
      specifications: insertProduct.specifications || null,
      createdAt: new Date(),
    };
    this.products.set(id, product);
    return product;
  }

  // Order operations
  async getOrder(orderId: string): Promise<Order | undefined> {
    return this.orders.get(orderId);
  }

  async getOrdersByUser(userId: number): Promise<Order[]> {
    return Array.from(this.orders.values()).filter(order => order.userId === userId);
  }

  async createOrder(insertOrder: InsertOrder): Promise<Order> {
    const id = this.currentOrderId++;
    const order: Order = {
      ...insertOrder,
      id,
      trackingNumber: insertOrder.trackingNumber || null,
      estimatedDelivery: insertOrder.estimatedDelivery || null,
      orderDate: new Date(),
    };
    this.orders.set(insertOrder.orderId, order);
    return order;
  }

  async updateOrderStatus(orderId: string, status: string): Promise<Order | undefined> {
    const order = this.orders.get(orderId);
    if (!order) return undefined;

    const updatedOrder = { ...order, status };
    this.orders.set(orderId, updatedOrder);
    return updatedOrder;
  }

  // Chat operations
  async getChatHistory(userId: number): Promise<Chat[]> {
    return Array.from(this.chats.values())
      .filter(chat => chat.userId === userId)
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }

  async createChat(insertChat: InsertChat): Promise<Chat> {
    const id = this.currentChatId++;
    const chat: Chat = {
      ...insertChat,
      id,
      context: insertChat.context || null,
      timestamp: new Date(),
    };
    this.chats.set(id, chat);
    return chat;
  }

  async getRecentChats(userId: number, limit: number): Promise<Chat[]> {
    return Array.from(this.chats.values())
      .filter(chat => chat.userId === userId)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }
}

export const storage = new MemStorage();
