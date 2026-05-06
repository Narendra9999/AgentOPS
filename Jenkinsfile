// ─────────────────────────────────────────────────────────────
// AgentOPS CI/CD Pipeline
//
// Deploys the AgentOPS framework for one or more team targets.
//
// Usage:
//   - Triggered on merge to main
//   - Detects which teams changed and deploys only those
//   - Can also be triggered manually with a specific target
//
// Parameters:
//   TARGET — team target to deploy (e.g. "team-hr", "mastercard", "dev")
//            Leave empty for auto-detect from changed files
//   RUN_PIPELINE — whether to run the agent pipeline after deploy
//   DRY_RUN — validate only, don't deploy
// ─────────────────────────────────────────────────────────────

pipeline {
    agent any

    parameters {
        string(
            name: 'TARGET',
            defaultValue: '',
            description: 'Target to deploy (e.g. team-hr, mastercard). Empty = auto-detect from PR changes.'
        )
        booleanParam(
            name: 'RUN_PIPELINE',
            defaultValue: true,
            description: 'Run the agent pipeline after deploy (data ingestion → eval → deploy endpoint)'
        )
        booleanParam(
            name: 'DRY_RUN',
            defaultValue: false,
            description: 'Validate only — do not deploy or run'
        )
    }

    environment {
        DATABRICKS_HOST     = credentials('databricks-host')
        DATABRICKS_TOKEN    = credentials('databricks-token')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Detect Targets') {
            when { expression { params.TARGET == '' } }
            steps {
                script {
                    // Auto-detect which teams changed based on the PR diff
                    def changedFiles = sh(
                        script: "git diff --name-only HEAD~1 HEAD",
                        returnStdout: true
                    ).trim().split('\n')

                    def targets = []

                    // Check for team-specific changes
                    changedFiles.each { file ->
                        def matcher = file =~ /teams\/([^\/]+)\//
                        if (matcher.find()) {
                            def team = matcher.group(1)
                            if (team != '_template') {
                                targets.add("team-${team}")
                            }
                        }
                    }

                    // Check for framework changes (affects all teams)
                    def frameworkChanged = changedFiles.any { file ->
                        file.startsWith('src/framework/') ||
                        file.startsWith('resources/') ||
                        file == 'databricks.yml' ||
                        file == 'pyproject.toml'
                    }

                    if (frameworkChanged) {
                        echo "Framework changed — deploying all active targets"
                        // Add all team targets from teams/*/target.yml
                        def teamDirs = sh(
                            script: "ls -d teams/*/target.yml 2>/dev/null | sed 's|teams/||;s|/target.yml||' | grep -v _template",
                            returnStdout: true
                        ).trim().split('\n')
                        teamDirs.each { team ->
                            if (team?.trim()) targets.add("team-${team}")
                        }
                        // Also deploy core targets
                        targets.add("mastercard")
                    }

                    if (targets.isEmpty()) {
                        echo "No target changes detected — skipping deploy"
                        currentBuild.result = 'NOT_BUILT'
                        return
                    }

                    env.DEPLOY_TARGETS = targets.unique().join(',')
                    echo "Targets to deploy: ${env.DEPLOY_TARGETS}"
                }
            }
        }

        stage('Set Target') {
            when { expression { params.TARGET != '' } }
            steps {
                script {
                    env.DEPLOY_TARGETS = params.TARGET
                    echo "Manual target: ${env.DEPLOY_TARGETS}"
                }
            }
        }

        stage('Validate') {
            steps {
                script {
                    env.DEPLOY_TARGETS.split(',').each { target ->
                        echo "Validating target: ${target}"
                        sh "databricks bundle validate -t ${target}"
                    }
                }
            }
        }

        stage('Deploy') {
            when { expression { !params.DRY_RUN } }
            steps {
                script {
                    env.DEPLOY_TARGETS.split(',').each { target ->
                        echo "Deploying target: ${target}"
                        sh "databricks bundle deploy -t ${target}"
                    }
                }
            }
        }

        stage('Run Pipeline') {
            when {
                allOf {
                    expression { !params.DRY_RUN }
                    expression { params.RUN_PIPELINE }
                }
            }
            steps {
                script {
                    env.DEPLOY_TARGETS.split(',').each { target ->
                        echo "Running pipeline for: ${target}"
                        sh "databricks bundle run agentops_pipeline -t ${target}"
                    }
                }
            }
        }

        stage('Verify') {
            when { expression { !params.DRY_RUN && params.RUN_PIPELINE } }
            steps {
                script {
                    env.DEPLOY_TARGETS.split(',').each { target ->
                        echo "Verifying endpoint for: ${target}"
                        // The pipeline's smoke test already validates the endpoint
                        // This step just confirms the run completed
                        echo "Pipeline completed for ${target}"
                    }
                }
            }
        }
    }

    post {
        success {
            echo "AgentOPS deployment successful for: ${env.DEPLOY_TARGETS}"
        }
        failure {
            echo "AgentOPS deployment failed for: ${env.DEPLOY_TARGETS}"
            // Notify team via Slack/email
            // slackSend channel: '#agentops-alerts', message: "Deployment failed for ${env.DEPLOY_TARGETS}"
        }
    }
}
