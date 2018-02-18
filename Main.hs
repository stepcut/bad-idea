{-# LANGUAGE DataKinds #-}
module Main where

import Control.Monad (replicateM)
-- import Control.Concurrent.STM
import Data.Attoparsec.ByteString (Result(..), IResult(..), parse, parseWith)
import Data.ByteString (hGetSome)
import qualified Data.ByteString as BS
-- import qualified Data.ByteString.Lazy as BS
import Data.MNIST
import Data.Word (Word8)
import Data.Vector (Vector, (!))
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector as Vector
import Graphics.Gloss hiding (Vector)
import Graphics.Gloss.Interface.Pure.Game hiding (Vector)
import Linear.Algebra (mult)
import Linear.Metric (dot)
import Linear.Vector (sumV, (^+^), (^-^), (^*), (*^),(^/))
import Linear.Matrix (transpose, (!*))
import System.Environment
import System.IO (openFile, IOMode(ReadMode))
import System.Random (randomIO)

windowRadius = 800

window :: Display
window = InWindow "bad idea" (windowRadius,windowRadius) (10,10)

background :: Color
background = white

fps :: Int
fps = 10

type Labels = MNIST UnsignedByte 1
type Digits = MNIST UnsignedByte 3

data World = World
 { index   :: Int
 , lbls    :: Labels
 , digits  :: Digits
 , network :: Network
 }


handleInput :: Event -> World -> World
handleInput _ w = w

update :: Float -> World -> World
update delta w = w { index = ((index w + 1) `mod` 60000) }

renderWeight :: Double -> Picture
renderWeight w = color (greyN (realToFrac (((sigmoid w) + 0.5) / 2))) (circleSolid 1)
--  color black (circleSolid 1)

renderNeuron :: Vector Double -> Picture
renderNeuron weights =
  let w = round (sqrt (fromIntegral (Vector.length weights)))
  in pictures [ translate (fromIntegral (i `mod` w)) (fromIntegral (i `div` w)) $ renderWeight (weights!i) | i <- [0.. pred (Vector.length weights)] ]

renderLayer :: Vector (Vector Double) -> Picture
renderLayer neurons =
  pictures [ translate 0 (fromIntegral $ i * 30) $ renderNeuron (neurons!i) | i <- [0 .. pred (Vector.length neurons)] ]

renderNetwork :: Network -> Picture
renderNetwork (Network _ layers biases _ _) = translate (-300) (-300) $ scale 1 1 $
  pictures [ translate (fromIntegral $ i*30) 0 $ renderLayer (layers!i) | i <- [0 .. pred (Vector.length layers)] ]

render :: World -> Picture
render (World i (UnsignedByteV1 _ lbls) (UnsignedByteV3 _ digits) network) =
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


main0 :: IO ()
main0 =
  do putStrLn "Parsing..."
     lblsFP <- BS.readFile "train-labels-idx1-ubyte"
     let (Done _ lbls) = parse pMNISTUnsignedByteV1 lblsFP
--     digitsFP <- BS.readFile "train-images-idx3-ubyte"
     h <- openFile "train-images-idx3-ubyte" ReadMode
     r <- parseWith (hGetSome h 1) pMNISTUnsignedByteV3 mempty
     n <- initNetwork [6, 20, 10]
     case r of
       (Done _ digits) ->
         do let world = World 0 lbls digits n
--            putStrLn "Waiting.."
--            getLine
            pure ()
            play window background fps world render handleInput update

main1 :: IO ()
main1 =
  do n <- initNetwork [28*28, 15, 10]
     play window background fps n renderNetwork (const id) (const id)


main3 :: IO ()
main3 =
  do n <- initNetwork [3, 2]
     print (weights n)
     print (biases n)

main = main3
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
  { sizes  :: [Int]
  , weights :: Vector (Vector (Vector Double)) -- Layer (Neuron (Weight Double))
  , biases  :: Vector (Vector Double)
  , activationFunction  :: Double -> Double
  , activationFunction' :: Double -> Double
  }
--  deriving (Eq, Ord, Read, Show)
{-
testNetwork0 :: Network
testNetwork0 = Network
  { sizes = [3]
  , weights = Vector.fromList (Vector.fromList 
-}

feedforward :: Network -> Vector Double -> Vector Double
feedforward network inputs = feedforward' (zip (Vector.toList $ weights network) (Vector.toList $ biases network))  inputs
  where
    feedforward' :: [(Vector (Vector Double), Vector Double)] -> Vector Double -> Vector Double
    feedforward' []         inputs = inputs
    feedforward' ((l,b):ls) inputs = feedforward' ls (Vector.map (activationFunction network) (((l !* inputs) ^+^ b)))


initNetwork :: [Int] -> IO Network
initNetwork szs =
  do bs <- Vector.fromList <$> (mapM (\s -> Vector.fromList <$> replicateM s randomIO) (tail szs))
     ws <- Vector.fromList <$>
       mapM (\(j,k) -> Vector.fromList <$> replicateM j (Vector.fromList <$> replicateM k randomIO)) (zip (tail szs) szs)
     pure $ Network { sizes = szs
                    , biases = bs
                    , weights = ws
                    , activationFunction = sigmoid
                    , activationFunction'= sigmoid'
                    }

networkAnd :: Network
networkAnd =
  Network { sizes = [2,1]
          , weights = Vector.fromList [ Vector.fromList [ Vector.fromList [ 10, 10 ] ] ]
          , biases  = Vector.fromList [ Vector.fromList [ (-12) ] ]
          , activationFunction = sigmoid
          , activationFunction' = sigmoid'
          }

networkNand :: Network
networkNand =
  Network { sizes = [2,1]
          , weights = Vector.fromList [ Vector.fromList [ Vector.fromList [ (-10), (-10) ] ] ]
          , biases  = Vector.fromList [ Vector.fromList [ 12 ] ]
          , activationFunction = sigmoid
          , activationFunction' = sigmoid'
          }

networkOr :: Network
networkOr =
  Network { sizes = [2,1]
          , weights = Vector.fromList [ Vector.fromList [ Vector.fromList [ 100, 100 ] ] ]
          , biases  = Vector.fromList [ Vector.fromList [ -90 ] ]
          , activationFunction = sigmoid
          , activationFunction' = sigmoid'
          }

networkXor :: Network
networkXor =
  Network { sizes = [2,2,1]
          , weights = Vector.fromList [ Vector.fromList [ Vector.fromList [ (-10), (-10) ], Vector.fromList [ 100, 100 ] ]
                                      , Vector.fromList [ Vector.fromList [ 10, 10 ] ]
                                      ]
          , biases  = Vector.fromList [ Vector.fromList [ 12, -90 ]
                                      , Vector.fromList [ -12 ]
                                      ]
          , activationFunction = sigmoid
          , activationFunction' = sigmoid'
          }

testNetwork :: Network -> [(Vector Double, Double)] -> IO ()
testNetwork nn testData =
  mapM_ (eval nn) testData
  where
    eval nn (inputs, expected) =
      let guess = feedforward nn inputs
      in putStrLn $ "inputs == " ++ show inputs ++ ", expected == " ++ show expected ++ ", guess == " ++ show guess

mainAnd :: IO ()
mainAnd =
  do let testData = [ (Vector.fromList [0, 0], 0)
                    , (Vector.fromList [0, 1], 0)
                    , (Vector.fromList [1, 0], 0)
                    , (Vector.fromList [1, 1], 1)
                    ]
       in testNetwork networkAnd testData

mainNand :: IO ()
mainNand =
  do let testData = [ (Vector.fromList [0, 0], 0)
                    , (Vector.fromList [0, 1], 0)
                    , (Vector.fromList [1, 0], 0)
                    , (Vector.fromList [1, 1], 1)
                    ]
       in testNetwork networkNand testData

mainOr :: IO ()
mainOr =
  do let testData = [ (Vector.fromList [0, 0], 0)
                    , (Vector.fromList [0, 1], 1)
                    , (Vector.fromList [1, 0], 1)
                    , (Vector.fromList [1, 1], 1)
                    ]
       in testNetwork networkOr testData

mainXor :: IO ()
mainXor =
  do let testData = [ (Vector.fromList [0, 0], 0)
                    , (Vector.fromList [0, 1], 1)
                    , (Vector.fromList [1, 0], 1)
                    , (Vector.fromList [1, 1], 1)
                    ]
       in testNetwork networkXor testData


{-
  do bs' <- mapM (const randomIO) (tail let)
     sizes bs = Vector.fromList (0:bs')
     ws <- Vector.fromList <$> mapM (\sz -> Vector.fromList <$> replicateM sz randomIO) sizes
     pure $ Network { biases = bs
                    , weights = ws
                    }
-}
--  Network { biases = 

sigmoid :: Double -> Double
sigmoid z =
  1.0 / (1.0 + exp (-z))

sigmoid' :: Double -> Double
sigmoid' z = (sigmoid z) * (1 - sigmoid z)

infixl 7 ^*^

(^*^) :: (Num a) => (Vector a) -> (Vector a) -> (Vector a)
a ^*^ b = Vector.zipWith (*) a b

z :: Vector (Vector Double) -> Vector -> Vector Double -> Vector Double
z weights a bias = (weights !* a) + bias

a :: (Double -> Double) -> Vector (Vector Double) -> Vector Double -> Vector Double
a f weights a' = Vector.map f (z weights a')

nabla_a_C :: Vector Double -> Vector Double -> Vector Double
nabla_a_C target activation = target ^-^ activation
{-
delta_output :: (Double -> Double) -> (Double -> Double) -> Vector (Vector Double) -> Vector Double -> Vector Double -> Vector Double
delta_output activationFunction activationFunction' weights activations targets =
  (nabla_a_C activations targets) ^*^ (Vector.map activationFunction (z 
-}

train :: Network -> Labels -> Digits -> Network
train network lbls digits = network
