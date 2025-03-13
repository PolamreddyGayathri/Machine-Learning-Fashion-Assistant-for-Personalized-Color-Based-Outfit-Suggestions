// import React, { useState, useEffect } from "react";
// import { Link, useLocation, useNavigate } from "react-router-dom";

// const Header = () => {
//   const navigate = useNavigate();
//   const location = useLocation();
  
//   // Ensure user state is managed correctly
//   const [user, setUser] = useState(() => localStorage.getItem("user"));

//   // Update user state whenever location changes (page navigation)
//   useEffect(() => {
//     setUser(localStorage.getItem("user"));
//   }, [location]);

//   // Handle logout and redirect to /home1
//   const handleLogout = () => {
//     localStorage.removeItem("user");
//     setUser(null);
//     navigate("/home1");  // ✅ Redirect to /home1 instead of "/"
//   };

//   // Ensure active link styling
//   const isActive = (path) => {
//     return location.pathname === path
//       ? "text-blue-600 border-b-2 border-blue-600"
//       : "text-gray-700 hover:text-blue-500";
//   };

//   // Set "About" link dynamically based on user authentication state
//   const aboutPath = user ? "/" : "/home1";

//   return (
//     <div className="p-4 h-[60px] w-full flex justify-between items-center bg-blue-50 top-0 sticky z-50 shadow-md">
//       {/* App Title */}
//       <Link to="/" className="text-2xl font-bold font-serif ml-5">
//         COLOR INSIGHT
//       </Link>

//       {/* Navigation Links */}
//       <div className="flex justify-center items-center gap-6 font-bold text-lg mr-6">
//         <Link to={aboutPath} className={`pb-1 ${isActive(aboutPath)}`}>
//           About
//         </Link>

//         {user ? (
//           <>
//             <Link to="/upload" className={`pb-1 ${isActive("/upload")}`}>
//               Upload
//             </Link>
//             <Link to="/cart" className={`pb-1 ${isActive("/cart")}`}>
//               Cart
//             </Link>
//             <Link to="/favourite" className={`pb-1 ${isActive("/favourite")}`}>
//               Favourite
//             </Link>
//             <button
//               onClick={handleLogout}
//               className="hover:underline underline-offset-2 text-red-500"
//             >
//               LogOut
//             </button>
//           </>
//         ) : (
//           <>
//             <Link to="/login" className={`pb-1 ${isActive("/login")}`}>
//               Login
//             </Link>
//           </>
//         )}
//       </div>
//     </div>
//   );
// };

// export default Header;

import React, { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

const Header = ({ user }) => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Ensure user state is managed correctly
  const [currentUser, setCurrentUser] = useState(user);

  // Update user state whenever location changes (page navigation)
  useEffect(() => {
    setCurrentUser(localStorage.getItem("user"));
  }, [location, user]);

  // Handle logout and redirect to /home1
  const handleLogout = () => {
    localStorage.removeItem("user");
    setCurrentUser(null);
    navigate("/home1");  // Redirect to /home1
  };

  // Ensure active link styling
  const isActive = (path) => {
    return location.pathname === path
      ? "text-blue-600 border-b-2 border-blue-600"
      : "text-gray-700 hover:text-blue-500";
  };

  // Set "About" link dynamically based on user authentication state
  const aboutPath = currentUser ? "/" : "/home1";

  return (
    <div className="p-4 h-[60px] w-full flex justify-between items-center bg-blue-50 top-0 sticky z-50 shadow-md">
      {/* App Title */}
      <Link to="/" className="text-2xl font-bold font-serif ml-5">
        COLOR INSIGHT
      </Link>

      {/* Navigation Links */}
      <div className="flex justify-center items-center gap-6 font-bold text-lg mr-6">
        <Link to={aboutPath} className={`pb-1 ${isActive(aboutPath)}`}>
          About
        </Link>

        {currentUser ? (
          <>
            <Link to="/upload" className={`pb-1 ${isActive("/upload")}`}>
              Upload
            </Link>
            <Link to="/cart" className={`pb-1 ${isActive("/cart")}`}>
              Cart
            </Link>
            <Link to="/favourite" className={`pb-1 ${isActive("/favourite")}`}>
              Favourite
            </Link>
            <button
              onClick={handleLogout}
              className="hover:underline underline-offset-2 text-red-500"
            >
              LogOut
            </button>
          </>
        ) : (
          <>
            <Link to="/login" className={`pb-1 ${isActive("/login")}`}>
              Login
            </Link>
          </>
        )}
      </div>
    </div>
  );
};

export default Header;
