{ nixpkgs ? import <nixpkgs> {}, compiler ? "default" }:

let

  inherit (nixpkgs) pkgs;

  f = { mkDerivation, attoparsec, attoparsec-binary, base, bytestring
      , data-binary-ieee754, stdenv, cabal-install, pipes-attoparsec, vector, random, stm
      , linear, gloss
      }:
      mkDerivation {
        pname = "bad-idea";
        version = "0.1.0.0";
        src = ./.;
        libraryHaskellDepends = [
          attoparsec attoparsec-binary base bytestring data-binary-ieee754 random stm linear gloss
        ] ++ (with pkgs.darwin.apple_sdk.frameworks; [ AGL Cocoa Foundation CoreData AppKit ]);
        buildTools = [ cabal-install ];
        homepage = "http://www.github.com/n-heptane-lab/bad-idea";
        description = "Another neural network / deep learning library";
        license = stdenv.lib.licenses.bsd3;
      };

  haskellPackages = if compiler == "default"
                       then pkgs.haskellPackages
                       else pkgs.haskell.packages.${compiler};

  drv = haskellPackages.callPackage f {};

in

  if pkgs.lib.inNixShell then drv.env else drv
