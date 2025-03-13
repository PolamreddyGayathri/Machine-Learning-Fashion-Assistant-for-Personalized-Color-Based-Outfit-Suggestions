import React from "react";
import AliceCarousel from "react-alice-carousel";
import "react-alice-carousel/lib/alice-carousel.css";
import { FaChevronLeft, FaChevronRight } from "react-icons/fa";
import { Link as ScrollLink } from "react-scroll"; // Import ScrollLink from react-scroll

const Home = () => {
  const items = [
    {
      color: "bg-red-500",
      title: "Red",
      description:
        "Red symbolizes passion, love, and energy. It’s bold, attention-grabbing, and associated with strength and power.",
      image:
        "https://media.istockphoto.com/id/484845976/photo/wardrobe-with-red-clothes-hanging-on-a-rack-nicely-arranged.jpg?s=2048x2048&w=is&k=20&c=5aduN3xNz2nljLWPxaYYPesXZYitsvbosP8_L1S7Y7I=",
    },
    {
      color: "bg-blue-500",
      title: "Blue",
      description:
        "Blue represents trust, intelligence, and calmness. It’s often used to create a sense of stability and professionalism.",
      image:
        "https://i.pinimg.com/originals/62/39/42/623942eaf6ee252820a4f18216f02959.jpg",
    },
    {
      color: "bg-green-500",
      title: "Green",
      description:
        "Green signifies nature, balance, and harmony. It’s linked to health, renewal, and eco-friendliness.",
      image:
        "https://img.freepik.com/premium-photo/stylish-green-closet_1017677-2663.jpg",
    },
    {
      color: "bg-yellow-500",
      title: "Yellow",
      description:
        "Yellow radiates happiness, optimism, and creativity. It’s vibrant and known to uplift moods.",
      image:
        "https://beautywithlily.com/wp-content/uploads/2019/08/yellowww.jpg",
    },
    {
      color: "bg-purple-500",
      title: "Purple",
      description:
        "Purple embodies luxury, creativity, and wisdom. It’s mysterious and often associated with royalty.",
      image:
        "https://img.freepik.com/premium-photo/closet-with-purple-shirts-purple-shirts-purple-one-that-says-other_854727-84457.jpg",
    },
  ];

  const carouselItems = items.map((item, index) => (
    <div
      key={index}
      className={`flex items-center justify-center ${item.color} w-screen h-[600px] text-white p-6 gap-10 shadow-lg rounded-xl hover:scale-105 transition-transform`}
    >
      <img
        className="rounded-3xl shadow-2xl h-[400px] w-[400px] hover:shadow-3xl transition-shadow"
        src={item.image}
        alt={item.title}
      />
      <div className="flex flex-col justify-center items-start bg-opacity-80 p-6 rounded-lg backdrop-blur-md hover:backdrop-blur-lg transition-all">
        <h2 className="text-4xl font-extrabold mb-4 animate-fade-in">
          {item.title}
        </h2>
        <p className="text-lg leading-relaxed max-w-md animate-fade-in-up">
          {item.description}
        </p>
      </div>
    </div>
  ));

  return (
    <>
      {/* Hero Section */}
      <div className="relative h-screen flex flex-col items-center justify-center bg-black text-white text-center overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-500 via-pink-500 to-red-500 animate-gradient-x opacity-70"></div>
        <h1 className="text-[100px] font-bold uppercase relative z-10 drop-shadow-xl animate-fade-in-down">
          Color Insight
        </h1>
        <h2 className="text-[50px] font-medium mt-6 relative z-10 animate-fade-in">
          What's Your Color?
        </h2>
        <p className="text-xl mt-4 relative z-10 animate-fade-in">
          Discover the meaning of colors and find your perfect style.
        </p>
        <ScrollLink
          to="about-section" // Target ID for the About Section
          smooth={true} // Enable smooth scrolling
          duration={500} // Scroll duration in milliseconds
          className="mt-8 px-8 py-3 bg-white text-black font-semibold rounded-full hover:bg-gray-200 transition-all relative z-10 animate-fade-in cursor-pointer"
        >
          Explore More
        </ScrollLink>
      </div>

      {/* Carousel */}
      <div className="relative w-full bg-gray-100 py-10">
        <AliceCarousel
          mouseTracking
          infinite
          items={carouselItems}
          responsive={{ 1024: { items: 1 } }}
          controlsStrategy="alternate"
          disableDotsControls
          renderPrevButton={() => (
            <button className="absolute left-4 top-1/2 transform -translate-y-1/2 bg-white p-3 shadow-xl rounded-full hover:scale-110 transition">
              <FaChevronLeft className="text-gray-700" />
            </button>
          )}
          renderNextButton={() => (
            <button className="absolute right-4 top-1/2 transform -translate-y-1/2 bg-white p-3 shadow-xl rounded-full hover:scale-110 transition">
              <FaChevronRight className="text-gray-700" />
            </button>
          )}
        />
      </div>

      {/* About Section */}
      <div
        id="about-section" // Add ID for scrolling target
        className="bg-gray-100 text-gray-800 py-20 px-6"
      >
        <h1 className="text-5xl font-bold text-center mb-12 animate-bounce">
          About Our Website
        </h1>
        <div className="flex flex-wrap justify-center gap-14 mt-10 max-w-7xl mx-auto">
          {["How It Works?", "Why Use Our Service?", "Our Mission?"].map(
            (title, index) => (
              <div
                key={index}
                className="bg-white shadow-lg rounded-xl p-8 hover:scale-105 transition-all opacity-90 hover:opacity-100 hover:bg-gray-200 max-w-sm w-full text-center"
              >
                <div className="text-4xl mb-4">
                  {index === 0 ? "🎨" : index === 1 ? "❤️" : "🚀"}
                </div>
                <h2 className="text-2xl font-semibold mb-4">{title}</h2>
                <p className="text-lg">
                  {[
                    "Using AI, we identify dominant colors in images and suggest clothing combinations that suit you best.",
                    "Match your outfits effortlessly and discover fashion inspiration based on your favorite colors.",
                    "Blending technology and fashion to create personalized style recommendations for you.",
                  ][index]}
                </p>
              </div>
            )
          )}
        </div>
      </div>

      {/* Footer Section */}
      
    </>
  );
};

export default Home;