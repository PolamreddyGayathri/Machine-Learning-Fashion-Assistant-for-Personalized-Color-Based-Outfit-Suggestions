import React, { useState } from "react";
import axios from "axios";
import data1 from "../Clothes.json";
import { CiShoppingCart } from "react-icons/ci";
import { FaRegHeart } from "react-icons/fa";

const Upload = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState("");
  const [clothes, setClothes] = useState([]);
  const [gender, setGender] = useState("");
  const [colorRecommendations, setColorRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const id = localStorage.getItem("id");
  const [skin, setSkin] = useState("");
  const [lip, setLip] = useState("");
  const [filter, setFilter] = useState("all");
  const [showColors, setShowColors] = useState(false);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (event) => {
    setLoading(true);
    event.preventDefault();

    if (!selectedImage) {
      alert("Please select an image before uploading.");
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload_image", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        const { response_image, response_eye, color_recommendations } = data;

        const lipToneMapping = {
          1: "spring",
          2: "summer",
          3: "autumn",
          4: "winter",
        };

        const skinTone =
          lipToneMapping[response_image?.result?.result] || "unknown";

        setColorRecommendations(color_recommendations);
        setSkin(skinTone);
        setLip(response_eye?.result?.result);
        handleClothes(color_recommendations);
      } else {
        console.error("Error:", JSON.stringify(data, null, 2));
        alert("Failed to upload the image. Please try again.");
      }
    } catch (error) {
      console.error("Upload error:", error.message);
      alert("An error occurred while uploading the image.");
    } finally {
      setLoading(false);
    }
  };

  const handleCartClick = async (item) => {
    try {
      const { data } = await axios.post("http://localhost:8080/user/addItem", {
        itemName: item?.name,
        itemPrice: item?.price,
        itemImage: item?.image,
        id: id,
      });
      if (data?.success) {
        alert("Item added successfully");
      }
      console.log("Item added:", data);
    } catch (error) {
      console.log(error);
    }
  };

  const handlefavClick = async (item) => {
    try {
      const { data } = await axios.post(
        "http://localhost:8080/user/addFavItem",
        {
          itemName: item?.name,
          itemPrice: item?.price,
          itemImage: item?.image,
          id: id,
        }
      );
      if (data?.success) {
        alert("Item added to favourite successfully");
      }
      console.log("Item added:", data);
    } catch (error) {
      console.log(error);
    }
  };

  const handleClothes = (color_recommendations) => {
    if (gender) {
      const genderData = data1.find((d) => d.gender === gender);

      if (!genderData) {
        console.error("No data found for the specified gender:", gender);
        return;
      }

      if (!genderData.colors || !Array.isArray(genderData.colors)) {
        console.error(
          "Colors data is missing or not in the correct format for the gender:",
          gender
        );
        return;
      }

      const normalizedColorRecommendations = color_recommendations.map(
        (color) =>
          color.startsWith("#")
            ? color.toLowerCase()
            : `#${color.toLowerCase()}`
      );

      const validColors = genderData.colors.filter((colorObj, index) => {
        const colorValue = colorObj.value || colorObj.hex;

        if (!colorValue) {
          console.warn(
            `Missing color value in colorObj at index ${index}:`,
            colorObj
          );
          return false;
        }

        if (!Array.isArray(colorObj.clothes)) {
          console.warn(
            `Invalid or missing 'clothes' in colorObj at index ${index}:`,
            colorObj
          );
          return false;
        }

        colorObj.value = colorValue;
        return true;
      });

      const filteredClothes = [];

      validColors.forEach((colorObj) => {
        const normalizedColorValue = colorObj.value.toLowerCase();

        if (normalizedColorRecommendations.includes(normalizedColorValue)) {
          filteredClothes.push(...colorObj.clothes);
        }
      });

      setClothes(filteredClothes);
    } else {
      console.error("Gender is not defined or invalid.");
    }
  };

  const filteredClothes = clothes.filter((item) => {
    if (filter === "all") return true;
    return item.category === filter;
  });

  return (
    <div
      className="min-h-screen flex justify-center items-center p-8"
      style={{
        background:
          "linear-gradient(135deg, #ff9a9e 0%, #fad0c4 50%, #a1c4fd 100%)",
      }}
    >
      <div className="flex flex-col items-center p-8 max-w-4xl mx-auto bg-white rounded-xl shadow-2xl">
        <div className="mb-8 text-center border-4 border-gray-300 h-[400px] w-[400px] flex justify-center items-center rounded-xl overflow-hidden">
          {preview ? (
            <img
              src={preview}
              className="h-full w-full object-cover"
              alt="Preview"
            />
          ) : (
            <span className="text-gray-500">No image selected</span>
          )}
        </div>

        <div className="flex justify-center items-center gap-6 mb-8">
          <label
            htmlFor="image"
            className="px-6 py-3 font-semibold text-white bg-violet-500 rounded-lg shadow-md cursor-pointer hover:bg-violet-600 transition-all duration-300"
          >
            Select Image
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="hidden"
            id="image"
          />
          <select
            className="font-serif font-semibold cursor-pointer p-3 rounded-lg outline-none bg-slate-200 hover:bg-slate-300 transition-all duration-300"
            value={gender}
            onChange={(e) => setGender(e.target.value)}
          >
            <option value="" disabled>
              Select Gender
            </option>
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>

        <button
          className="mt-6 bg-violet-500 text-white px-8 py-3 rounded-lg font-semibold hover:bg-violet-600 transition-all duration-300 shadow-lg"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? "Uploading..." : "Submit"}
        </button>

        <div className="w-full flex flex-col justify-center items-center mt-10">
          <div className="flex flex-row justify-center items-center text-2xl font-bold gap-20 whitespace-nowrap">
            {skin && (
              <span>
                Skin Tone: <span className="font-extrabold">{skin}</span>
              </span>
            )}
            {lip && (
              <span>
                Eye Colour: <span className="font-extrabold">{lip}</span>
              </span>
            )}
          </div>

          {colorRecommendations.length > 0 && (
            <div className="w-full flex flex-col items-center mt-6">
              {showColors && (
                <div className="w-full flex flex-wrap justify-center gap-4">
                  {colorRecommendations.map((color, index) => (
                    <div
                      key={index}
                      className="w-16 h-16 rounded-lg shadow-md"
                      style={{ backgroundColor: color }}
                    ></div>
                  ))}
                </div>
              )}

              <button
                className="mt-4 bg-pink-500 text-white px-6 py-3 rounded-lg text-lg font-bold hover:bg-pink-600 transition-all duration-300 shadow-lg"
                onClick={() => setShowColors(!showColors)}
              >
                {showColors ? "Hide Color Palette" : "Show Color Palette"}
              </button>
            </div>
          )}
        </div>

        {clothes?.length > 0 && (
          <div className="mt-10 w-full">
            <h2 className="text-2xl font-bold mb-6 text-center">
              Recommended Clothes
            </h2>
            <div className="flex justify-center items-center gap-4 mb-8">
              {["all", "traditional", "casual", "party", "formal"].map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`px-6 py-2 rounded-lg font-semibold ${
                    filter === f
                      ? "bg-violet-500 text-white"
                      : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  } transition-all duration-300`}
                >
                  {f.charAt(0).toUpperCase() + f.slice(1)}
                </button>
              ))}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredClothes?.map((item, index) => (
                <div
                  key={index}
                  className="p-6 bg-gray-100 rounded-xl shadow-md hover:shadow-lg transition-all duration-300"
                >
                  <img
                    src={item.image}
                    alt={item.name}
                    className="h-64 w-full object-cover mb-4 rounded-lg"
                  />
                  <h3 className="text-xl font-semibold mb-2">{item.name}</h3>
                  <p className="text-gray-600 mb-4">{item.price}</p>
                  <div className="flex justify-center items-center gap-4">
                    <button
                      onClick={() => handleCartClick(item)}
                      className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-all duration-300"
                    >
                      <CiShoppingCart size={24} />
                    </button>
                    <button
                      onClick={() => handlefavClick(item)}
                      className="px-6 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-all duration-300"
                    >
                      <FaRegHeart size={24} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Upload;