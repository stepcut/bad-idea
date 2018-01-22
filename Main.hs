{-# LANGUAGE DataKinds #-}
module Main where

import Control.Monad (replicateM)
-- import Control.Concurrent.STM
import Data.Attoparsec.ByteString.Lazy (Result(..), parse)
import Data.ByteString (hGetSome)
import qualified Data.ByteString.Lazy as BS
import Data.MNIST
import Data.Word (Word8)
import Data.Vector (Vector, (!))
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector as Vector
import Graphics.Gloss hiding (Vector)
import Graphics.Gloss.Interface.Pure.Game hiding (Vector)
import System.Environment
import System.IO (openFile, IOMode(ReadMode))
import System.Random (randomIO)

windowRadius = 800

window :: Display
window = InWindow "bad idea" (windowRadius,windowRadius) (10,10)

background :: Color
background = white

fps :: Int
fps = 5

data World = World
 { index  :: Int
 , lbls   :: MNIST UnsignedByte 1
 , digits :: MNIST UnsignedByte 3
 }
 deriving Show

handleInput :: Event -> World -> World
handleInput _ w = w

update :: Float -> World -> World
update delta w = w { index = ((index w + 1) `mod` 60000) }

render :: World -> Picture
render (World i (UnsignedByteV1 _ lbls) (UnsignedByteV3 _ digits)) =
   pictures [ translate 0 (200) $ text        (show i)
            , translate 0 (100) $ text        (show (lbls U.! i))
            , scale 4 4         $ renderDigit (digits ! i)
            ]
  where
    grey' :: Word8 -> Color
    grey' w = greyN $ 1 - ((fromIntegral w) / 255)
    renderDigit :: U.Vector Word8 -> Picture
    renderDigit digit = pictures $
      do x <- [0..27]
         y <- [0..27]
         pure $ translate x (-y) $ color (grey' (digit U.! (((truncate y) * 28) + (truncate x)))) $ circleSolid 0.5
--         pure $ translate x (-y) $ color (grey' (digit ! (truncate y) U.! (truncate x))) $ circleSolid 0.5


main :: IO ()
main =
  do putStrLn "Parsing..."
     lblsFP <- BS.readFile "train-labels-idx1-ubyte"
     let (Done _ lbls) = parse pMNISTUnsignedByteV1 lblsFP
     digitsFP <- BS.readFile "train-images-idx3-ubyte"
     case parse pMNISTUnsignedByteV3 digitsFP of
       (Done _ digits) ->
         do let world = World 0 lbls digits
--            putStrLn "Waiting.."
--            getLine
            pure ()
            play window background fps world render handleInput update

{-
     train fp

train :: FilePath -> IO ()
train fp =
  do h <- openFile fp ReadMode
     bs <- hGetSome h 4
     case parse magicWord bs of
      (Done _ (size, dims)) ->
        do bs <- hGetSome h (4 * (fromIntegral dims))
           case parse (sizes dims) bs of
            (Done _ szs) ->
              do print (size, dims, szs)
-}
data Network = Network
  { biases :: Vector Double
  , weights :: Vector (Vector (Vector Double))
  }
  deriving (Eq, Ord, Read, Show)


feedforward :: Network -> Double -> Vector Double
feedforward network a =
 undefined

initNetwork :: [Int] -> IO Network
initNetwork szs =
  do bs <- Vector.fromList <$> (mapM (const randomIO) (tail szs))
     ws <- Vector.fromList <$>
       mapM (\(j,k) -> Vector.fromList <$> replicateM j (Vector.fromList <$> replicateM k randomIO)) (zip (tail szs) szs)
     pure $ Network { biases = bs
                    , weights = ws
                    }

{-
  do bs' <- mapM (const randomIO) (tail let)
     sizes bs = Vector.fromList (0:bs')
     ws <- Vector.fromList <$> mapM (\sz -> Vector.fromList <$> replicateM sz randomIO) sizes
     pure $ Network { biases = bs
                    , weights = ws
                    }
-}
--  Network { biases = 

sigmoid z =
  1.0 / (1.0 + exp (-z))


