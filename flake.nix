{
  description = "Molecular integrals over Gaussian basis functions using sympy";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          overlay = import ./nix/overlay.nix;
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ overlay ];
          };
        in
        {
          packages.default = pkgs.python3.pkgs.sympleints;

          formatter = pkgs.nixpkgs-fmt;
        }) // {
      overlays.default = import ./nix/overlay.nix;
    };
}
