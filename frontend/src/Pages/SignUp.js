// import React, { useState } from "react";
// import { Link, useNavigate } from "react-router-dom";
// import axios from "axios";

// const SignUp = () => {
//   const navigate = useNavigate();
//   const [name, setName] = useState("");
//   const [email, setEmail] = useState("");
//   const [password, setPassword] = useState("");
//   const URL = "http://localhost:8080";

//   const handelRegister = async () => {
//     if (!name || !email || !password) {
//       return alert("Please fill all fields!!");
//     }
//     try {
//       const { data } = await axios.post(`${URL}/user/register`, {
//         name: name,
//         email: email,
//         password: password,
//       });
//       if (data?.success) {
//         alert("Successfully registered!!");
//         navigate("/login");
//       }
//     } catch (error) {
//       console.log(error);
//     }
//   };

//   return (
//     <div className="h-screen flex items-center justify-center bg-gradient-to-br from-[#FFD6E8] via-[#D1E8FF] to-[#E0FFDA] relative overflow-hidden">
//       <div className="absolute inset-0 bg-gradient-to-r from-[#F472B6] to-[#A78BFA] opacity-20 blur-3xl -z-10 animate-[gradient-move_6s_infinite_alternate] bg-[length:200%_200%]" />
//       <div className="bg-white bg-opacity-60 shadow-2xl rounded-2xl flex flex-col p-8 max-w-md w-full relative overflow-hidden animate-bounce-in-down backdrop-blur-md">
//         <h1 className="text-5xl font-extrabold mb-8 text-center text-transparent bg-clip-text bg-gradient-to-r from-[#A78BFA] to-[#F472B6] animate-gradient">
//           REGISTER
//         </h1>

//         <input
//           className="p-4 mb-6 rounded-lg border border-gray-300 outline-none shadow-md focus:shadow-lg transition-all duration-200 animate-fade-in"
//           placeholder="Name"
//           value={name}
//           onChange={(e) => setName(e.target.value)}
//         />
//         <input
//           className="p-4 mb-6 rounded-lg border border-gray-300 outline-none shadow-md focus:shadow-lg transition-all duration-200 animate-fade-in delay-100"
//           placeholder="Email"
//           value={email}
//           onChange={(e) => setEmail(e.target.value)}
//         />
//         <input
//           className="p-4 mb-6 rounded-lg border border-gray-300 outline-none shadow-md focus:shadow-lg transition-all duration-200 animate-fade-in delay-200"
//           placeholder="Password"
//           value={password}
//           onChange={(e) => setPassword(e.target.value)}
//         />

//         <button
//           className="p-4 rounded-lg bg-gradient-to-r from-[#A78BFA] to-[#F472B6] text-white font-bold hover:opacity-90 shadow-md hover:shadow-lg transition-all duration-200"
//           onClick={handelRegister}
//         >
//           Register
//         </button>

//         <h1 className="font-bold text-xl text-center mt-6 text-[#6B7280] animate-fade-in-up delay-300">Or</h1>
//         <h1 className="font-semibold text-lg text-center mt-4 animate-fade-in-up delay-400">
//           Already have an account?{' '}
//           <Link
//             to="/login"
//             className="text-[#A78BFA] hover:text-[#F472B6] hover:underline underline-offset-4 transition-all duration-200"
//           >
//             Login
//           </Link>
//         </h1>
//       </div>
//     </div>
//   );
// };

// export default SignUp;

import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { FaEye, FaEyeSlash } from "react-icons/fa";

const SignUp = () => {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const URL = "http://localhost:8080";

  const isValidPassword = (password) => {
    return /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/.test(password);
  };

  const handleRegister = async () => {
    if (!name || !email || !password) {
      return alert("Please fill all fields!!");
    }

    if (!isValidPassword(password)) {
      return alert(
        "Password must be at least 8 characters long and include an uppercase letter, a lowercase letter, a number, and a special character."
      );
    }

    try {
      const { data } = await axios.post(`${URL}/user/register`, { name, email, password });
      if (data?.success) {
        alert("Successfully registered!");
        navigate("/login");
      }
    } catch (error) {
      console.log(error.response?.data || error.message);
      alert("Registration failed, please try again.");
    }
  };

  return (
    <div className="h-screen flex items-center justify-center bg-gradient-to-br from-[#FFD6E8] via-[#D1E8FF] to-[#E0FFDA] relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-[#F472B6] to-[#A78BFA] opacity-20 blur-3xl -z-10 animate-[gradient-move_6s_infinite_alternate] bg-[length:200%_200%]" />
      <div className="bg-white bg-opacity-60 shadow-2xl rounded-2xl flex flex-col p-8 max-w-md w-full relative overflow-hidden animate-[bounce-in-down_1s_ease-out] backdrop-blur-md">
        <h1 className="text-5xl font-extrabold mb-8 text-center text-transparent bg-clip-text bg-gradient-to-r from-[#A78BFA] to-[#F472B6] animate-gradient">
          REGISTER
        </h1>

        <input
          className="p-4 mb-6 rounded-lg border border-gray-300 outline-none shadow-md focus:shadow-lg transition-all duration-200 animate-fade-in"
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <input
          className="p-4 mb-6 rounded-lg border border-gray-300 outline-none shadow-md focus:shadow-lg transition-all duration-200 animate-fade-in delay-100"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <div className="relative w-full mb-6">
          <input
            type={showPassword ? "text" : "password"}
            className="p-4 rounded-lg border border-gray-300 outline-none shadow-md focus:shadow-lg transition-all duration-200 w-full"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value.trim())}
          />
          <span
            className="absolute right-4 top-1/2 transform -translate-y-1/2 cursor-pointer text-gray-600"
            onClick={() => setShowPassword(!showPassword)}
          >
            {showPassword ? <FaEyeSlash /> : <FaEye />}
          </span>
        </div>

        <button
          className="p-4 rounded-lg bg-gradient-to-r from-[#A78BFA] to-[#F472B6] text-white font-bold hover:opacity-90 shadow-md hover:shadow-lg transition-all duration-200"
          onClick={handleRegister}
        >
          REGISTER
        </button>

        <h1 className="font-bold text-xl text-center mt-6 text-[#6B7280] animate-fade-in-up delay-300">Or</h1>
        <h1 className="font-semibold text-lg text-center mt-4 animate-fade-in-up delay-400">
          Already have an account?{' '}
          <Link
            to="/login"
            className="text-[#A78BFA] hover:text-[#F472B6] hover:underline underline-offset-4 transition-all duration-200"
          >
            Login
          </Link>
        </h1>
      </div>
    </div>
  );
};

export default SignUp;
