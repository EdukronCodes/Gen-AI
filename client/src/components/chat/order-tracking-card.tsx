import { useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

interface OrderTrackingCardProps {
  orderId: string;
}

export default function OrderTrackingCard({ orderId }: OrderTrackingCardProps) {
  const { data: order, isLoading } = useQuery({
    queryKey: ["/api/orders", orderId],
    enabled: !!orderId,
  });

  if (isLoading) {
    return (
      <div className="ml-11">
        <Card className="bg-gray-50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-3">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-6 w-16" />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-3 w-24" />
              </div>
              <div className="flex justify-between">
                <Skeleton className="h-3 w-32" />
                <Skeleton className="h-3 w-24" />
              </div>
              <div className="flex justify-between">
                <Skeleton className="h-3 w-24" />
                <Skeleton className="h-3 w-36" />
              </div>
            </div>
            <div className="mt-4">
              <Skeleton className="h-8 w-full" />
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!order) {
    return (
      <div className="ml-11">
        <Card className="bg-gray-50">
          <CardContent className="p-4 text-center">
            <p className="text-gray-600">Order not found.</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "delivered":
        return "bg-green-100 text-green-800";
      case "shipped":
        return "bg-blue-100 text-blue-800";
      case "processing":
        return "bg-yellow-100 text-yellow-800";
      case "cancelled":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getProgressPercentage = (status: string) => {
    switch (status.toLowerCase()) {
      case "placed":
        return 25;
      case "processing":
        return 50;
      case "shipped":
        return 75;
      case "delivered":
        return 100;
      default:
        return 0;
    }
  };

  return (
    <div className="ml-11">
      <Card className="bg-gray-50">
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-gray-900">Order #{order.orderId}</h4>
            <Badge className={getStatusColor(order.status)}>
              {order.status.charAt(0).toUpperCase() + order.status.slice(1)}
            </Badge>
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Order Date:</span>
              <span className="text-gray-900">
                {new Date(order.orderDate).toLocaleDateString()}
              </span>
            </div>
            {order.estimatedDelivery && (
              <div className="flex justify-between">
                <span className="text-gray-600">Estimated Delivery:</span>
                <span className="text-gray-900">
                  {new Date(order.estimatedDelivery).toLocaleDateString()}
                </span>
              </div>
            )}
            {order.trackingNumber && (
              <div className="flex justify-between">
                <span className="text-gray-600">Tracking Number:</span>
                <span className="text-blue-600 font-mono">{order.trackingNumber}</span>
              </div>
            )}
          </div>

          {/* Order Progress */}
          <div className="mt-4">
            <div className="flex items-center justify-between text-xs text-gray-600 mb-2">
              <span>Order Placed</span>
              <span>Processing</span>
              <span>Shipped</span>
              <span>Delivered</span>
            </div>
            <div className="relative">
              <div className="h-2 bg-gray-200 rounded-full">
                <div
                  className="h-2 bg-green-500 rounded-full transition-all duration-300"
                  style={{ width: `${getProgressPercentage(order.status)}%` }}
                ></div>
              </div>
              <div className="absolute top-0 left-0 w-3 h-3 bg-green-500 rounded-full -mt-0.5"></div>
              <div className="absolute top-0 left-1/3 w-3 h-3 bg-green-500 rounded-full -mt-0.5"></div>
              <div className="absolute top-0 left-2/3 w-3 h-3 bg-green-500 rounded-full -mt-0.5"></div>
              <div className="absolute top-0 right-0 w-3 h-3 bg-gray-300 rounded-full -mt-0.5"></div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
