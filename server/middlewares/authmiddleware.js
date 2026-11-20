import jwt from "jsonwebtoken"

const authncheck=(req,res,next)=>
{
    const token=req.cookies.accessToken;
    if(!token){
       return res.status(401).json(
            {
                message:"no token bruh!!"
            }
        )
    }

    try{

        const decoded=jwt.verify(token,process.env.JWT_SECRET);

        req.userId=decoded.userId;
        
        next();
    }catch(err)
    {
        return res.status(401).json(
            {
                message:"expired or invalid token "
            }
        )
    }
}

export default authncheck;