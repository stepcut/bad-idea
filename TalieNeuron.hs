module Main where

import Data.Vector ((!), Vector)
import qualified Data.Vector as Vector
import Debug.Trace (trace)
import Graphics.Gloss hiding (Vector)
import Graphics.Gloss.Interface.Pure.Game hiding (Vector)
import Linear.Algebra (mult)
import Linear.Metric (dot)
import Linear.Vector (sumV, (^+^),(^*), (*^))
import Linear.Matrix ((!*))
import System.Environment
import System.Exit (exitSuccess)
import System.Random (newStdGen, randomIO, randomRIO, randomRs)

-- | data type for neuron
data Neuron = Neuron
  { weights              :: Vector Float
  , bias                 :: Float
  , activationFunction   :: Float -> Float
  , activationFunction' :: Float -> Float
  }

-- | sigmoid function
sigmoid :: Float -> Float
sigmoid t = 1/(1 + exp (-t))

sigmoid' :: Float -> Float
sigmoid' z = (sigmoid z) * (1 - sigmoid z)

-- | heaviside step function
heavisideStep :: Float -> Float
heavisideStep x = if x < 0 then 0 else 1

-- | evaluate neuron
evalNeuron :: Neuron -> Vector Float -> Float
evalNeuron (Neuron weights bias activationFunction _) inputs =
  activationFunction ((dot inputs weights) + bias)

-- | make perceptron
mkPerceptron :: Int -> IO  Neuron
mkPerceptron numberInputs =
  do gen <- newStdGen
     pure (Neuron { weights = Vector.fromList (take numberInputs (randomRs (-0.1, 0.1) gen))
                  , bias  = 0
                  , activationFunction = heavisideStep
                  , activationFunction' = undefined
                  })

-- | make sigmoid neuron
mkSigmoid :: Int -> IO  Neuron
mkSigmoid numberInputs =
  do gen <- newStdGen
     pure (Neuron { weights = Vector.fromList (take numberInputs (randomRs (-0.1, 0.1) gen))
                  , bias  = 0.01
                  , activationFunction = sigmoid
                  , activationFunction' = sigmoid'
                  })

-- | frames per second
fps :: Int
fps = 60

-- | window radius
windowRadius = 800

-- | the window
window :: Display
window = InWindow "bad idea" (windowRadius,windowRadius) (10,10)

-- | window background color
background :: Color
background = white

-- | record which contains the state of the `World`
data World = World
    { neuron :: Neuron
    , trainingData :: [(Vector Float, Float)]
    , index :: Int
    }

-- | create the initial state of the `World`
mkWorld :: IO World
mkWorld =
  do n <- mkPerceptron 2
     pure $ World { neuron       = n
                  , trainingData = []
                  , index        = 0
                  }

-- | go out perceptron
goOut :: Neuron
goOut =  Neuron { weights = Vector.fromList [6, 2, 2]
                  , bias = (-5)
                  , activationFunction  = heavisideStep
                  , activationFunction' = undefined
  }

-- | Perceptron Training function
train :: (Vector Float, Float) -> Neuron -> Neuron
train (inputs, target) neuron =
  let guess    = evalNeuron neuron inputs
      error    = target - guess
      weights' = (weights neuron) ^+^ inputs ^* error
      bias'    = (bias neuron) + error
  in neuron { weights = weights'
            , bias    = bias'
            }

-- | Training function 2
trains :: [(Vector Float, Float)] -> Neuron -> Neuron
trains [] neuron = neuron
trains (td : tds) neuron = trains tds (train td neuron)


h :: Neuron -> Vector Float -> Float
h neuron inputs = (dot inputs (weights neuron)) + (bias neuron)

deltaW :: Neuron -> (Vector Float, Float) -> (Vector Float, Float)
deltaW neuron (inputs, target) =
  let alpha  = 0.4
      err    = target - (evalNeuron neuron inputs)
      gPrime = (activationFunction' neuron) (h neuron inputs)
  in (alpha * err * gPrime *^ inputs, 0.1 * err * gPrime)

trainDeltaOnline :: (Vector Float, Float) -> Neuron -> Neuron
trainDeltaOnline td n =
  let (dw, db) = deltaW n td
      weights' = weights n ^+^ dw
      bias'    = bias n + db
  in n { weights = weights'
       , bias    = bias'
       }


-- | Recognize AND function
andState :: IO World
andState =
  do perceptron <- mkPerceptron 2
     let trainingset = [ (Vector.fromList [0, 0], 0)
                       , (Vector.fromList [0, 1], 0)
                       , (Vector.fromList [1, 0], 0)
                       , (Vector.fromList [1, 1], 1)
                       ]
     pure ( World { neuron       = perceptron
                  , trainingData = trainingset
                  , index        = 0
              } )

mateState :: IO World
mateState =
  do perceptron <- mkPerceptron 2
     g1 <- newStdGen
     g2 <- newStdGen
     let xs = randomRs (0, 1) g1
         ys = randomRs (0, 1) g2
         td = take 150 $ zipWith (\x y -> (Vector.fromList[x,y], isAcceptable x y)) xs ys
     pure $ World { neuron       = perceptron
                  , trainingData = td
                  , index        = 0
                  }
       where
         isAcceptable hot wealth =
           if (wealth > (-0.5 * hot) + 0.5) then 1 else 0

sigmoidState :: IO World
sigmoidState =
  do neuron <- mkSigmoid 2
     g1 <- newStdGen
     g2 <- newStdGen
     let xs = randomRs (0, 1) g1
         ys = randomRs (0, 1) g2
         td = take 150 $ zipWith (\x y -> (Vector.fromList[x,y], isAcceptable x y)) xs ys
     pure $ World { neuron       = neuron
                  , trainingData = td
                  , index        = 0
                  }
       where
         isAcceptable hot wealth =
           if (wealth > (-0.5 * hot) + 0.5) then 1 else 0

-- | Create training cycle
trainN :: Int -> [(Vector Float, Float)] -> Neuron -> Neuron
trainN 0 _ neuron = neuron
trainN n td neuron = trainN (n-1) td (trains td neuron)

-- | given a state of the `World` create a `Picture`
render :: World -> Picture
render world =
  color black $ pictures [ xLabel
                         , yLabel
                         , boundary
                         , (drawDataPoints (neuron world) (trainingData world))
                         , drawNeuron (neuron world)
                         ]
  where
    xLabel = translate (-20) (-370) $ scale 0.1 0.1 $ Text "Hotness"
    yLabel = translate (-360) 10 $ rotate (-90) $ scale 0.1 0.1 $ Text "$$$"
    boundary = lineLoop [ (-350, -350)
                        , (-350, 350)
                        , (350, 350)
                        , (350, -350)
                        ]

drawNeuron :: Neuron -> Picture
drawNeuron (Neuron weights bias _ _) =
  line [ (toScreen 0, toScreen (x1 0))
       , (toScreen 1, toScreen (x1 1))
       ]
  where
    w0 = weights!0
    w1 = weights!1
    x1 x0 = - (x0 * (w0 / w1)) - (bias / w1)

drawO :: Picture
drawO = circle 5

drawX :: Picture
drawX = pictures [ line [(-5, -5), (5, 5)]
                 , line [(-5, 5), (5, -5)]
                 ]

translate' :: Float -> Float -> Picture -> Picture
translate' x y p  = translate (toScreen x) (toScreen y) p

toScreen :: Float -> Float
toScreen z = 700*z - 350

drawDataPoint :: Neuron -> (Vector Float, Float) -> Picture
drawDataPoint neuron (inputs, target) =
  let guess = evalNeuron neuron inputs
      c = if (abs (target - guess) < 0.5)
          then color green
          else color red

  in translate' (inputs!0) (inputs!1) $
      if target >= 0.5 then c drawO else c drawX

drawDataPoints :: Neuron -> [(Vector Float, Float)] -> Picture
drawDataPoints neuron dps = (pictures (map (drawDataPoint neuron) dps ))


-- | An `Event` occured, how does the `World` change?
handleInput :: Event -> World -> World
handleInput event world = world

{-
-- | time `delta` has passed, how has the `World` changed?
update :: Float -> World -> World
update delta world =
  world { neuron = trains (trainingData world) (neuron world) }
-}

-- | time `delta` has passed, how has the `World` changed?
update :: Float -> World -> World
update delta world =
  world { neuron = train ((trainingData world)!!(index world)) (neuron world)
        , index  = ((index world) + 1) `mod` (length (trainingData world))
        }

-- | time `delta` has passed, how has the `World` changed?
updateDeltaOnline :: Float -> World -> World
updateDeltaOnline _delta world =
  world { neuron = trainDeltaOnline ((trainingData world)!!(index world)) (neuron world)
        , index  = ((index world) + 1) `mod` (length (trainingData world))
        }

-- | applications starts here
main :: IO ()
main =
  do -- perceptron <t- mkPerceptron 2
     world@(World neuron trainingData _) <- sigmoidState
--     let neuron' = trainN 20 trainingData neuron
     play window background fps world render handleInput updateDeltaOnline
{-
     let neuron' = neuron
     putStrLn "Untrained"
     putStrLn ("0 && 0 = " ++ show (evalNeuron neuron' (Vector.fromList[0, 0])))
     putStrLn ("0 && 1 = " ++ show (evalNeuron neuron' (Vector.fromList[0, 1])))
     putStrLn ("1 && 0 = " ++ show (evalNeuron neuron' (Vector.fromList[1, 0])))
     putStrLn ("1 && 1 = " ++ show (evalNeuron neuron' (Vector.fromList[1, 1])))

     putStrLn "Trained"
     putStrLn ("0 && 0 = " ++ show (evalNeuron neuron' (Vector.fromList[0, 0])))
     putStrLn ("0 && 1 = " ++ show (evalNeuron neuron' (Vector.fromList[0, 1])))
     putStrLn ("1 && 0 = " ++ show (evalNeuron neuron' (Vector.fromList[1, 0])))
     putStrLn ("1 && 1 = " ++ show (evalNeuron neuron' (Vector.fromList[1, 1])))
-}
--     world <- mkWorld
--     print (evalNeuron goOut (Vector.fromList [ 0, 1, 1]))
--     print (evalNeuron goOut (Vector.fromList [ 1, 0, 0]))

