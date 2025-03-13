import axios from "axios";
import React, { useEffect, useState } from "react";

const Cart = () => {
  const id = localStorage.getItem("id");
  const [cart, setCart] = useState([]);

  const getData = async () => {
    try {
      const { data } = await axios.post(
        "http://localhost:8080/user/singleUser",
        { id: id }
      );

      if (data?.success) {
        setCart(data?.user?.cart);
      }
    } catch (error) {
      console.log(error);
    }
  };

  const handleRemoveFromCart = async (itemId) => {
    try {
      await axios.post("http://localhost:8080/user/delete", {
        userID: id,
        itemID: itemId,
      });

      getData();
    } catch (error) {
      console.error("Error removing item from cart:", error);
    }
  };

  useEffect(() => {
    getData();
  }, []);

  return (
    <div
      className="min-h-screen p-8"
      style={{
        background:
          "linear-gradient(135deg, #f5f7fa, #c3cfe2)",
      }}
    >
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold text-center mb-8 text-black">
          Your Cart
        </h2>
        {cart.length === 0 ? (
          <p className="text-center text-2xl text-black mt-20">
            Your cart is empty.
          </p>
        ) : (
          <>
            <div className="flex justify-between items-center mb-8">
              <h1 className="text-2xl font-bold text-black">
                Total Items: {cart?.length}
              </h1>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {cart?.map((item) => (
                <div
                  key={item.id}
                  className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300"
                >
                  <div className="w-full h-64 flex items-center justify-center overflow-hidden p-4 bg-gray-100 border border-gray-200 rounded-t-lg">
                    <img
                      src={item?.itemImage}
                      alt={item.itemName}
                      className="max-w-full max-h-full object-contain"
                    />
                  </div>
                  <div className="p-6">
                    <h3 className="text-xl font-semibold mb-2 text-black">
                      {item.itemName}
                    </h3>
                    <p className="text-gray-600 mb-4">${item.itemPrice}</p>
                    <button
                      onClick={() => handleRemoveFromCart(item?._id)}
                      className="w-full bg-red-500 text-white py-2 px-4 rounded-lg hover:bg-red-600 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-red-500"
                    >
                      Remove from Cart
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Cart;