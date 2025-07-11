import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface ProductRecommendationsProps {
  userId: number;
}

export default function ProductRecommendations({ userId }: ProductRecommendationsProps) {
  const { data: recommendations, isLoading } = useQuery({
    queryKey: ["/api/recommendations", userId],
  });

  if (isLoading) {
    return (
      <div className="ml-11">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="bg-gray-50">
              <CardContent className="p-4">
                <Skeleton className="w-full h-32 rounded-md mb-3" />
                <Skeleton className="h-4 w-3/4 mb-2" />
                <Skeleton className="h-3 w-full mb-2" />
                <div className="flex items-center justify-between">
                  <Skeleton className="h-4 w-16" />
                  <Skeleton className="h-8 w-20" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (!recommendations || recommendations.length === 0) {
    return (
      <div className="ml-11">
        <Card className="bg-gray-50">
          <CardContent className="p-4 text-center">
            <p className="text-gray-600">No recommendations available at this time.</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="ml-11">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {recommendations.slice(0, 3).map((product: any) => (
          <Card key={product.id} className="bg-gray-50 hover:shadow-md transition-shadow">
            <CardContent className="p-4">
              <img
                src={product.imageUrl}
                alt={product.name}
                className="w-full h-32 object-cover rounded-md mb-3"
              />
              <h4 className="font-semibold text-gray-900 mb-1">{product.name}</h4>
              <p className="text-gray-600 text-sm mb-2">{product.description}</p>
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold text-gray-900">${product.price}</span>
                <Button
                  size="sm"
                  className="bg-primary hover:bg-primary/90 text-white"
                >
                  View Details
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
