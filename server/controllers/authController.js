import bcrypt from "bcryptjs"
import User from "../models/userSchema.js"
import jwt from "jsonwebtoken"


export const registerUser = async(req,res)=>
{
    const email=req.body.email;
    const password=req.body.password;
    const fullName=req.body.fullName;

    if(!email || !password || !fullName)
    {
        return res.sendStatus(400);
    }
    //check for duplicate email
    //check for strong password 
    const isUsed = await User.findOne({email:email});

    if(isUsed){
        return res.status(409).json({
            success: false,
            message: "Email already exists"
    });}

    try{

        const hashedpass= await bcrypt.hash(password,12);

        const newUser = await User.create({
        email,                                      //shorthand e:e to e since both are same words
        password : hashedpass,
        name : fullName

        
    })

    return res.sendStatus(201);
    //redirect to login

    }

    catch(err)
    {
        console.log(err);
        return res.sendStatus(500);
    }

    
}


export const loginUser = async(req,res)=>//check for already logged in

{
    
    const email=req.body.email;
    const password=req.body.password;
   

    if(!email || !password)
    {
        return res.sendStatus(400);
    }
    
    const curruser=await User.findOne({email:email});
    try{
       
        if(!curruser)return res.status(401).json({
            success:false,
            error:"wrong credentials"
        });

        const passMatch=await bcrypt.compare(password,curruser.password);

        if(!passMatch)return res.status(401).json(
            {
                success:false,
                error:"wrong password"
            }
        )

        

    }catch(e)
    {
        return res.status(500).json({
            success: false,
            error: "Internal server error"
        });
    }
    //jwt sign 
    const token = jwt.sign({
            userID : curruser._id
        },
        process.env.JWT_SECRET,
        {
            expiresIn :"15m"//how long jwt will last
    });

    //store cookie on client side.
    res.cookie("accessToken",token ,{
        secure : process.env.NODE_ENV === "production",
        httpOnly:true,
        sameSite:"lax",
        maxAge: 15 * 60 * 1000 //15 min (how long should browser keep cookie)
    })


    return res.status(200).json({
            success:true,
            message: "Login successful",
        user: {
            id: curruser._id,
            email: curruser.email
        }})
}


export const logoutUser = async(req,res)=>
{

}


export const userData = async(req,res)=>
{
    return res.send("<h1>hello</h1>")
}

