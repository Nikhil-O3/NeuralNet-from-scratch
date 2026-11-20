import express from "express"
import {registerUser , loginUser , logoutUser , userData } from "../controllers/authController.js"
import authncheck from '../middlewares/authmiddleware.js';

const router=express.Router();

// router.use(authncheck);

router.post("/register",registerUser);

router.post("/login",loginUser);

router.post("/logout",logoutUser);

router.get("/me",authncheck,userData);

export default router;