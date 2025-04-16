// // import React from "react";
// // import { BrowserRouter, Routes, Route } from "react-router-dom";
// // import Login from "./Pages/Login";
// // import SignUp from "./Pages/SignUp";
// // import Home from "./Pages/Home";
// // import Home1 from "./Pages/Home1";
// // import Header from "./Components/Header";
// // import Footer from "./Components/Footer";
// // import Upload from "./Pages/Upload";
// // import Form from "./Pages/Form";
// // import Cart from "./Pages/Cart";
// // import Favourite from "./Pages/Favourite";

// // const App = () => {
// //   return (
// //     <BrowserRouter>
// //       <div className="flex flex-col min-h-screen">
// //         <Header />
// //         <main className="flex-grow">
// //           <Routes>
// //             <Route path="/" element={<Home />} />
// //             <Route path="/home1" element={<Home1 />} />
// //             <Route path="/login" element={<Login />} />
// //             <Route path="/signup" element={<SignUp />} />
// //             <Route path="/upload" element={<Upload />} />
// //             <Route path="/form" element={<Form />} />
// //             <Route path="/cart" element={<Cart />} />
// //             <Route path="/favourite" element={<Favourite />} />
// //           </Routes>
// //         </main>
// //         <Footer />
// //       </div>
// //     </BrowserRouter>
// //   );
// // };

// // export default App;

// import React, { useState, useEffect } from "react";
// import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
// import Login from "./Pages/Login";
// import SignUp from "./Pages/SignUp";
// import Home from "./Pages/Home";
// import Home1 from "./Pages/Home1";
// import HeaderBeforeLogin from "./Components/HeaderBeforeLogin";
// import HeaderAfterLogin from "./Components/HeaderAfterLogin";
// import Footer from "./Components/Footer";
// import Upload from "./Pages/Upload";
// import Form from "./Pages/Form";
// import Cart from "./Pages/Cart";
// import Favourite from "./Pages/Favourite";

// const App = () => {
//   const [isAuthenticated, setIsAuthenticated] = useState(false);

//   // Check login status when the page loads
//   useEffect(() => {
//     const user = localStorage.getItem("user");
//     setIsAuthenticated(!!user);
//   }, []);

//   return (
//     <BrowserRouter>
//       <div className="flex flex-col min-h-screen">
//         {/* Show different headers before and after login */}
//         {isAuthenticated ? <HeaderAfterLogin setIsAuthenticated={setIsAuthenticated} /> : <HeaderBeforeLogin />}

//         <main className="flex-grow">
//           <Routes>
//             {/* Show Home1 before login, Home after login */}
//             <Route path="/" element={isAuthenticated ? <Home /> : <Home1 />} />
//             <Route path="/home" element={isAuthenticated ? <Home /> : <Navigate to="/" />} />
//             <Route path="/login" element={<Login setIsAuthenticated={setIsAuthenticated} />} />
//             <Route path="/signup" element={<SignUp />} />
//             <Route path="/upload" element={<Upload />} />
//             <Route path="/form" element={<Form />} />
//             <Route path="/cart" element={<Cart />} />
//             <Route path="/favourite" element={<Favourite />} />
//           </Routes>
//         </main>

//         <Footer />
//       </div>
//     </BrowserRouter>
//   );
// };

// export default App;

import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import Login from "./Pages/Login";
import SignUp from "./Pages/SignUp";
import Home from "./Pages/Home";
import Home1 from "./Pages/Home1";
import Header from "./Components/Header";
import Footer from "./Components/Footer";
import Upload from "./Pages/Upload";
import Form from "./Pages/Form";
import Cart from "./Pages/Cart";
import Favourite from "./Pages/Favourite";
import Camera from "./Pages/Camera";
import Profile from "./Pages/Profile";

const App = () => {
  const location = useLocation();
  const [user, setUser] = useState(localStorage.getItem("user"));

  useEffect(() => {
    setUser(localStorage.getItem("user"));
  }, [location]);

  return (
    <div className="flex flex-col min-h-screen">
      {/* Show Header dynamically based on login state */}
      <Header user={user} />
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={user ? <Home /> : <Home1 />} />
          <Route path="/home1" element={<Home1 />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/form" element={<Form />} />
          <Route path="/cart" element={<Cart />} />
          <Route path="/camera" element={<Camera />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/favourite" element={<Favourite />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
};

const AppWrapper = () => (
  <BrowserRouter>
    <App />
  </BrowserRouter>
);

export default AppWrapper;
