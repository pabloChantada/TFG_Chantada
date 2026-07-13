# La memoria usa fontspec -> obliga a XeLaTeX (pdf_mode = 5).
$pdf_mode = 5;

# Glosario y lista de acrónimos: ejecutar makeglossaries entre pasadas.
add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
add_cus_dep('acn', 'acr', 0, 'run_makeglossaries');

sub run_makeglossaries {
  if ( $silent ) {
    system 'makeglossaries', '-q', $_[0];
  }
  else {
    system 'makeglossaries', $_[0];
  };
}

$clean_ext = "acn acr alg bbl glg glo gls ist nav out snm xdv";
