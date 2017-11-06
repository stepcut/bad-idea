module Main where

import Data.Vector ((!), Vector)
import qualified Data.Vector as Vector
import Debug.Trace (trace)
import Graphics.Gloss hiding (Vector)
import Graphics.Gloss.Interface.Pure.Game hiding (Vector)
import Linear.Algebra (mult)
import Linear.Metric (dot)
import Linear.Vector (sumV, (^+^),(^*) )
import Linear.Matrix ((!*))
import System.Environment
import System.Exit (exitSuccess)
import System.Random (newStdGen, randomIO, randomRIO, randomRs)

-- | data type for neuron
data Neuron = Neuron
  { weights             :: Vector Float
   , bias               :: Float
   , activationFunction :: Float -> Float
  }

-- | heaviside step function
heavisideStep :: Float -> Float
heavisideStep x = if x<0 then (-1) else 1

-- | evaluate neuron
evalNeuron :: Neuron -> Vector Float -> Float
evalNeuron (Neuron weights bias activationFunction) inputs =
  activationFunction ((dot inputs weights) + bias)

-- | make perceptron
mkPerceptron :: Int -> IO  Neuron
mkPerceptron numberInputs =
  do gen <- newStdGen
     pure (Neuron { weights = Vector.fromList (take numberInputs (randomRs (-0.1, 0.1) gen))
                 , bias  = 0
                 , activationFunction = heavisideStep
                 })

-- | frames per second
fps :: Int
fps = 120

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
             }

-- | create the initial state of the `World`
-- mkWorld :: IO World
-- mkWorld = pure World

-- | go out perceptron
goOut :: Neuron
goOut =  Neuron { weights = Vector.fromList [6, 2, 2]
                  , bias = (-5)
                  , activationFunction = heavisideStep
  }

-- | Training function
train :: (Vector Float, Float) -> Neuron -> Neuron
train (inputs, target) neuron =
  let guess = evalNeuron neuron inputs
      error = target - guess
      weights' = (weights neuron) ^+^ inputs ^* error
      bias' = (bias neuron) + error
  in neuron { weights = weights'
            , bias = bias'
            }

-- | Training function 2
trains :: [(Vector Float, Float)] -> Neuron -> Neuron
trains [] neuron = neuron
trains (td : tds) neuron = trains tds (train td neuron)

-- | Recognize AND function
andState :: IO World
andState =
  do perceptron <- mkPerceptron 2
     let trainingset = [ (Vector.fromList [(-1), (-1)], (-1))
                       , (Vector.fromList [(-1),   1],  (-1))
                       , (Vector.fromList [1,    (-1)], (-1))
                       , (Vector.fromList [1,      1],    1)
                       ]
     pure ( World { neuron       = perceptron
                  , trainingData = trainingset
              } )

-- | Create training cycle
trainN :: Int -> [(Vector Float, Float)] -> Neuron -> Neuron
trainN 0 _ neuron = neuron
trainN n td neuron = trainN (n-1) td (trains td neuron)

-- | given a state of the `World` create a `Picture`
render :: World -> Picture
render world = color black (pictures [ (circle 300)
                                     , (arc 200 340 10)
                                     , (translate 100 100 (circle 20))
                                     , (translate (-100) 100 (circle 20))
                                     ])

-- | An `Event` occured, how does the `World` change?
handleInput :: Event -> World -> World
handleInput event world = world

-- | time `delta` has passed, how has the `World` changed?
update :: Float -> World -> World
update delta world = world

-- | applications starts here
main :: IO ()
main =
  do -- world <- mkWorld
--     play window background fps world render handleInput update
     perceptron <- mkPerceptron 2
--     print (evalNeuron goOut (Vector.fromList [ 0, 1, 1]))
--     print (evalNeuron goOut (Vector.fromList [ 1, 0, 0]))
     (World neuron trainingData) <- andState
     let neuron' = neuron
     putStrLn "Untrained"
     putStrLn ("-1 && -1 = " ++ show (evalNeuron neuron' (Vector.fromList[(-1), (-1)])))
     putStrLn ("-1 && 1 = " ++ show (evalNeuron neuron' (Vector.fromList[(-1), (1)])))
     putStrLn ("1 && -1 = " ++ show (evalNeuron neuron' (Vector.fromList[(1), (-1)])))
     putStrLn ("1 && 1 = " ++ show (evalNeuron neuron' (Vector.fromList[(1), (1)])))
     putStrLn "Trained"
     let neuron' = trainN 20 trainingData neuron
     putStrLn ("-1 && -1 = " ++ show (evalNeuron neuron' (Vector.fromList[(-1), (-1)])))
     putStrLn ("-1 && 1 = " ++ show (evalNeuron neuron' (Vector.fromList[(-1), (1)])))
     putStrLn ("1 && -1 = " ++ show (evalNeuron neuron' (Vector.fromList[(1), (-1)])))
     putStrLn ("1 && 1 = " ++ show (evalNeuron neuron' (Vector.fromList[(1), (1)])))
