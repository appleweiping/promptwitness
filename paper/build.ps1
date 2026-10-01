# Build main.pdf with pdflatex + bibtex (latexmk needs Perl, which is not installed here).
Set-Location $PSScriptRoot
$log = "build.log"
"" | Out-File -Encoding utf8 $log
pdflatex -interaction=nonstopmode main.tex *>> $log
bibtex main *>> $log
pdflatex -interaction=nonstopmode main.tex *>> $log
pdflatex -interaction=nonstopmode main.tex *>> $log
$errors = Select-String -Path main.log -Pattern "^!" -Context 0, 2
$undef = Select-String -Path main.log -Pattern "(Citation|Reference) .* undefined"
$pages = Select-String -Path main.log -Pattern "Output written on main.pdf \((\d+) pages"
"errors: $($errors.Count)"; $errors | Select-Object -First 8 | ForEach-Object { $_.Line; $_.Context.PostContext }
"undefined: $($undef.Count)"; $undef | Select-Object -First 20 | ForEach-Object { $_.Line }
if ($pages) { $pages.Line }
