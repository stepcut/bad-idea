module Data.MNIST where

import Control.Monad
-- import Data.Attoparsec
import Data.Attoparsec.Binary
import Data.Attoparsec.ByteString
import Data.Int (Int32)
import Data.Word
import Data.Vector (Vector, fromList)

data Size
  = UnsignedByte
  | SignedByte
  | Short
  | Int
  | Float
  | Double
    deriving (Eq, Ord, Read, Show)

size :: Parser Size
size =
  do s <- anyWord8
     case s of
       0x08 -> pure UnsignedByte
       0x09 -> pure SignedByte
       0x0B -> pure Short
       0x0C -> pure Int
       0x0D -> pure Float
       0x0E -> pure Double
       _    -> fail $ show s ++ "is not a valid size in the magic word"

-- | (data size, number of dimensions)
magicWord :: Parser (Size, Word8)
magicWord =
  do word8 0
     word8 0
     s <- size
     dim <- anyWord8
     pure (s, dim)

sizes :: Word8 -> Parser [Word32]
sizes dims = replicateM (fromIntegral dims) anyWord32be

pV2UnsignedByte :: Word32 -> Word32 -> Parser (Vector (Vector Word8))
pV2UnsignedByte x y =
  fromList <$> replicateM (fromIntegral y) (pV1UnsignedByte x)

pV1UnsignedByte :: Word32 -> Parser (Vector Word8)
pV1UnsignedByte c =
  fromList <$> replicateM (fromIntegral c) anyWord8
