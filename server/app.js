import express from "express"
import cors from "cors";//to handle development operation calls between node and react.
import authRouter from './routes/authentication.js';
import cookieParser from "cookie-parser";

const app = express();

app.use(express.json());//for all purpose . req body "{}"to {}
app.use(cors());//development cross origin handler
app.use(cookieParser());//?for authn authr , set cookier direcly by browser with res.cookie ( no react involment)

//handle requests
app.get("/",(req,res)=>
{
    return res.send("res").status(200);
})

app.use("/api/auth",authRouter);

export default app; 