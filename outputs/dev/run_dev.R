# need to call this for VS Code debugger to work
readRenviron(".Renviron")

options(golem.app.prod = FALSE)
golem::detach_all_attached()
golem::document_and_reload()

# need to explicitly print for VS Code debugger to work
app <- run_app()
print(app)
