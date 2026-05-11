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

                        // For team targets, overlay team config onto agent config
                        // so RegisterModel packages the team-specific system prompt,
                        // guardrails, scorers, and evaluation settings.
                        def teamDir = ""
                        if (target.startsWith("team-")) {
                            // Resolve team directory from target name
                            // team-platform-eng → platform-engineering, team-data-gov → data-governance, etc.
                            def teamDirs = sh(
                                script: """grep -l 'team_dir' teams/*/target.yml | while read f; do
                                    dir=\$(dirname \$f | xargs basename)
                                    tgt=\$(grep -A1 'targets:' \$f | tail -1 | sed 's/:.*//' | tr -d ' ')
                                    if [ "\$tgt" = "${target}" ]; then echo \$dir; fi
                                done""",
                                returnStdout: true
                            ).trim()
                            if (teamDirs) {
                                teamDir = teamDirs
                                echo "Overlaying team config: teams/${teamDir}/config.yaml"
                                sh "cp src/agent_development/agent/config.yaml src/agent_development/agent/config.yaml.bak"
                                sh "cp teams/${teamDir}/config.yaml src/agent_development/agent/config.yaml"

                                // Overlay team scorers (domain + llm_judge) if they exist
                                sh """
                                    if [ -d teams/${teamDir}/scorers/domain ] && ls teams/${teamDir}/scorers/domain/*.yaml 2>/dev/null; then
                                        cp teams/${teamDir}/scorers/domain/*.yaml src/agent_development/agent_evaluation/evaluation/scorers/domain/
                                        echo "Copied team domain scorers"
                                    fi
                                    if [ -d teams/${teamDir}/scorers/llm_judge ] && ls teams/${teamDir}/scorers/llm_judge/*.yaml 2>/dev/null; then
                                        cp teams/${teamDir}/scorers/llm_judge/*.yaml src/agent_development/agent_evaluation/evaluation/scorers/llm_judge/
                                        echo "Copied team LLM judge scorers"
                                    fi
                                """
                            }
                        }

                        sh "databricks bundle deploy -t ${target}"

                        // Restore original config after deploy
                        if (teamDir) {
                            sh "mv src/agent_development/agent/config.yaml.bak src/agent_development/agent/config.yaml"
                        }
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
